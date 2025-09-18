package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/creack/pty"
)

func main() {
	fmt.Println("Starting PTY example...")
	
	cmd := exec.Command("/bin/bash")

	ptmx, err := pty.Start(cmd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pty.Start error: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = ptmx.Close() }()

}