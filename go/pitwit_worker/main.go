package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
)

type TelemetryPacket struct {
	CarID        string             `json:"car_id"`
	Lap          int                `json:"lap"`
	LapDistanceM float64            `json:"lap_distance_m"`
	SpeedKPH     float64            `json:"speed_kph"`
	Throttle     float64            `json:"throttle"`
	Brake        float64            `json:"brake"`
	TireAgeLaps  float64            `json:"tire_age_laps"`
	TrackTempC   float64            `json:"track_temp_c"`
	AirTempC     float64            `json:"air_temp_c"`
	Compound     string             `json:"compound"`
	TrackStatus  string             `json:"track_status"`
	TsNS         int64              `json:"ts_ns"`
	Extras       map[string]float64 `json:"extras,omitempty"`
}

type InferenceResult struct {
	CarID           string  `json:"car_id"`
	Lap             int     `json:"lap"`
	PitRisk         float64 `json:"pit_risk"`
	TireDegradation float64 `json:"tire_degradation"`
	Confidence      float64 `json:"confidence"`
	ModelLatencyNS  int64   `json:"model_latency_ns"`
	TsNS            int64   `json:"ts_ns"`
}

type Frame struct {
	Telemetry TelemetryPacket `json:"telemetry"`
	Inference InferenceResult `json:"inference"`
	Status    string          `json:"status"`
	Index     int             `json:"index"`
}

type job struct {
	index  int
	packet TelemetryPacket
}

func main() {
	workers := flag.Int("workers", runtime.NumCPU(), "number of parallel scoring workers")
	flag.Parse()

	if *workers < 1 {
		*workers = 1
	}

	jobs := make(chan job, *workers*4)
	results := make(chan Frame, *workers*4)

	var wg sync.WaitGroup
	for workerID := 0; workerID < *workers; workerID++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range jobs {
				results <- score(item)
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	scanner := bufio.NewScanner(os.Stdin)
	index := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var packet TelemetryPacket
		if err := json.Unmarshal(line, &packet); err != nil {
			fmt.Fprintf(os.Stderr, "invalid json at line %d: %v\n", index+1, err)
			os.Exit(1)
		}
		if packet.TsNS == 0 {
			packet.TsNS = time.Now().UnixNano()
		}
		jobs <- job{index: index, packet: packet}
		index++
	}
	close(jobs)

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "read input: %v\n", err)
		os.Exit(1)
	}

	encoder := json.NewEncoder(os.Stdout)
	for frame := range results {
		if err := encoder.Encode(frame); err != nil {
			fmt.Fprintf(os.Stderr, "write output: %v\n", err)
			os.Exit(1)
		}
	}
}

func score(item job) Frame {
	start := time.Now()
	p := item.packet
	degradation := math.Min(1.0, math.Max(0.0, p.TireAgeLaps/35.0))
	cheapStop := 0.0
	switch p.TrackStatus {
	case "YELLOW", "VSC", "SC", "SAFETY_CAR":
		cheapStop = 1.0
	}
	score := p.TireAgeLaps*0.08 + cheapStop*0.65 + degradation*0.75
	pitRisk := sigmoid(score)
	confidence := math.Max(pitRisk, 1.0-pitRisk)
	status := "GREEN"
	if pitRisk >= 0.75 {
		status = "RED"
	} else if pitRisk >= 0.45 {
		status = "AMBER"
	}

	return Frame{
		Telemetry: p,
		Inference: InferenceResult{
			CarID:           p.CarID,
			Lap:             p.Lap,
			PitRisk:         pitRisk,
			TireDegradation: degradation,
			Confidence:      confidence,
			ModelLatencyNS:  time.Since(start).Nanoseconds(),
			TsNS:            p.TsNS,
		},
		Status: status,
		Index:  item.index,
	}
}

func sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Exp(-value))
}
