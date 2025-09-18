// cmd/ailog/main.go
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
)

func main() {
	var (
		logDir     string
		noRaw      bool
		initCols   int
		initRows   int
		defaultCMD = []string{"claude", "chat"} // used if no args provided
	)
	flag.StringVar(&logDir, "logdir", ".ai-cli-log", "directory to write logs")
	flag.BoolVar(&noRaw, "no-raw", false, "do not set your terminal to raw mode")
	flag.IntVar(&initCols, "cols", 0, "initial terminal columns (0 = use current)")
	flag.IntVar(&initRows, "rows", 0, "initial terminal rows (0 = use current)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] -- <command> [args...]\n", os.Args[0])
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n  %s -- claude code --resume last\n", os.Args[0])
	}
	flag.Parse()

	argv := flag.Args()
	if len(argv) == 0 {
		argv = defaultCMD
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = os.Environ()

	// Start child in a PTY.
	ptmx, err := pty.Start(cmd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pty.Start error: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = ptmx.Close() }()

	// Initial window size.
	if initCols == 0 || initRows == 0 {
		if w, h, e := term.GetSize(int(os.Stdout.Fd())); e == nil {
			if initCols == 0 {
				initCols = w
			}
			if initRows == 0 {
				initRows = h
			}
		}
	}
	if initCols > 0 && initRows > 0 {
		_ = pty.Setsize(ptmx, &pty.Winsize{Cols: uint16(initCols), Rows: uint16(initRows)})
	}

	// Propagate window resizes to the PTY.
	winch := make(chan os.Signal, 1)
	signal.Notify(winch, syscall.SIGWINCH)
	go func() {
		for range winch {
			if w, h, e := term.GetSize(int(os.Stdout.Fd())); e == nil {
				_ = pty.Setsize(ptmx, &pty.Winsize{Cols: uint16(w), Rows: uint16(h)})
			}
		}
	}()
	// Kick once.
	winch <- syscall.SIGWINCH

	// Put *your* terminal into raw mode for natural TTY behavior.
	var oldState *term.State
	if !noRaw && term.IsTerminal(int(os.Stdin.Fd())) {
		if s, e := term.MakeRaw(int(os.Stdin.Fd())); e == nil {
			oldState = s
			defer func() { _ = term.Restore(int(os.Stdin.Fd()), oldState) }()
		}
	}

	// Prepare prompt-only logs.
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "log dir error: %v\n", err)
	}
	start := time.Now()
	base := start.Format("20060102-150405")
	prefix := fmt.Sprintf("%s-%s", base, slug(filepath.Base(argv[0])))

	promptsTxtPath := filepath.Join(logDir, prefix+".prompts.txt")
	promptsJSONLPath := filepath.Join(logDir, prefix+".prompts.jsonl")

	promptsTxt, err := os.Create(promptsTxtPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "prompts txt log error: %v\n", err)
	}
	defer func() {
		if promptsTxt != nil {
			_ = promptsTxt.Sync()
			_ = promptsTxt.Close()
		}
	}()

	promptsJSONL, err := os.Create(promptsJSONLPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "prompts jsonl log error: %v\n", err)
	}
	defer func() {
		if promptsJSONL != nil {
			_ = promptsJSONL.Sync()
			_ = promptsJSONL.Close()
		}
	}()

	// Graceful shutdown on Ctrl-C: close logs & PTY so buffers flush.
	intc := make(chan os.Signal, 1)
	signal.Notify(intc, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-intc
		if promptsTxt != nil {
			_ = promptsTxt.Sync()
			_ = promptsTxt.Close()
		}
		if promptsJSONL != nil {
			_ = promptsJSONL.Sync()
			_ = promptsJSONL.Close()
		}
		_ = ptmx.Close() // signal child via hangup on master side
		os.Exit(130)     // 130 = SIGINT exit code convention
	}()

	copyErrCh := make(chan error, 2)

	// PTY -> screen (we do NOT log assistant/output in this mode).
	go func() {
		_, e := io.Copy(os.Stdout, ptmx)
		copyErrCh <- e
	}()

	// stdin -> PTY; on Enter (CR or LF), write prompt to logs immediately.
	go func() {
		reader := bufio.NewReader(os.Stdin)
		var buf bytes.Buffer

		for {
			r, _, rerr := reader.ReadRune()
			if rerr != nil {
				copyErrCh <- rerr
				return
			}
			// Always forward the keystroke to the child.
			if _, werr := ptmx.WriteString(string(r)); werr != nil {
				copyErrCh <- werr
				return
			}

			// End-of-prompt on CR or LF (macOS sends '\r' in raw mode).
			if r == '\r' || r == '\n' {
				msg := strings.TrimSpace(buf.String())
				if msg != "" {
					// Plain text
					if promptsTxt != nil {
						fmt.Fprintln(promptsTxt, msg)
						_ = promptsTxt.Sync()
					}
					// JSONL
					if promptsJSONL != nil {
						rec := map[string]string{"role": "user", "content": msg}
						b, _ := json.Marshal(rec)
						fmt.Fprintln(promptsJSONL, string(b))
						_ = promptsJSONL.Sync()
					}
				}
				buf.Reset()
				continue
			}

			// Accumulate the current line.
			buf.WriteRune(r)
		}
	}()

	// Wait for either stream to end, then wait for the child to exit.
	_ = <-copyErrCh
	_ = cmd.Wait()

	fmt.Fprintf(os.Stderr, "[ai-log] prompts(txt):   %s\n", promptsTxtPath)
	fmt.Fprintf(os.Stderr, "[ai-log] prompts(jsonl): %s\n", promptsJSONLPath)
}

// slug creates a safe filename bit from a command name.
func slug(s string) string {
	var b bytes.Buffer
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '-' || c == '_' {
			b.WriteByte(c)
		} else {
			b.WriteByte('-')
		}
	}
	return b.String()
}
