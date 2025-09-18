#!/usr/bin/env python3
import subprocess, sys
import datetime

FMT = r"%i %M %S"   # stepid elapsed starttime

def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def main(jobid: str):
    try:
        # List steps for the job (portable): shows <jobid>.<stepid>
        out = run(["squeue", "-s", "-j", jobid, "-h", "-o", FMT])
    except subprocess.CalledProcessError as e:
        print(f"Error: cannot query steps via squeue: {e}")
        sys.exit(1)

    if not out:
        print(f"No steps found for job {jobid} (maybe not running yet?).")
        return

    running = []
    for line in out.splitlines():
        # print(line)
        # Example line: "710588.batch RUNNING 00:17:42 2025-09-08T22:10:03"
        parts = line.split(maxsplit=3)
        step_id, elapsed, start = parts
        step_id = step_id.strip()
        if not step_id.startswith(f"{jobid}."):
            continue
        if 'extern' in step_id:
            continue
        rec = dict(step_id=step_id,  elapsed=elapsed, start=start)
        # print(f"step_id: {step_id}, elapsed: {elapsed}, start: {start}")
        running.append(rec)
        # (running if state == "RUNNING" else others).append(rec)

    for r in running:
        step_id = r['step_id']
        elapsed = r['elapsed']
        start = r['start']
        elapsed_time = datetime.datetime.strptime(elapsed, "%M:%S")
        elapsed_time_ts = elapsed_time.timestamp()
        print(f"{step_id} has been running for {elapsed} (started {start}). (elapsed_time_ts: {elapsed_time_ts})")
        if elapsed_time_ts > 4 * 60: # timeout 4 min
            print(f"🔴 {step_id} has been running for {elapsed} (started {start}).")
            # Check if the folder exists...

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <jobid>")
        sys.exit(1)
    main(sys.argv[1])