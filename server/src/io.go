package main

import (
	"bufio"
	"encoding/binary"
	"encoding/csv"
	"io"
	"os"
	"strconv"
	"strings"
)

var outWriter = bufio.NewWriter(os.Stdout)

func parseFloatList(s string) []float64 {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err == nil {
			out = append(out, val)
		}
	}
	return out
}

func parseIntList(s string) []int {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		val, err := strconv.Atoi(strings.TrimSpace(p))
		if err == nil {
			out = append(out, val)
		}
	}
	return out
}

func parseStringList(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseComplexList(s string) []complex128 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	vals := make([]complex128, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		c, err := strconv.ParseComplex(p, 128)
		if err != nil {
			// handle error or skip
			continue
		}
		vals = append(vals, c)
	}
	return vals
}

// ----------- CSV writer -----------

func writeCSV(w io.Writer, data []float64) error {
	cw := csv.NewWriter(w)
	record := make([]string, len(data))
	for i, v := range data {
		record[i] = strconv.FormatFloat(v, 'g', -1, 64)
	}
	if err := cw.Write(record); err != nil {
		return err
	}
	cw.Flush()
	return cw.Error()
}

func writeBinary(w io.Writer, data []float64) error {
	// write length prefix
	if err := binary.Write(outWriter, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}
	// write float64s
	if err := binary.Write(outWriter, binary.LittleEndian, data); err != nil {
		return err
	}
	return outWriter.Flush() // <-- flush buffer to stdout
}
