"""
read jsonl file from --dir (artifacts/runs/20260613-122521/logs/ppo_metrics.jsonl)
and extract "raw_score_mean" for every step 
and "league_average_score" for every 10 steps
and plot them in a graph with x-axis as steps and y-axis as scores
"""

import argparse
import json
from matplotlib import pyplot as plt

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", type=str, required=True, help="directory of the jsonl file") # artifacts/runs/20260613-122521
  parser.add_argument("--output", type=str, help="output file for the plot", default=None) # --dir + "/logs/curve.png"
  args = parser.parse_args()
  output = args.output or args.dir + "/logs/curve.png"
  dir = args.dir + "/logs/ppo_metrics.jsonl"
  steps = []
  step_tens = []
  raw_scores = []
  league_scores = []
  with open(dir, "r") as f:
    for line in f:
      data = json.loads(line)
      steps.append(data["update"])
      raw_scores.append(data["raw_score_mean"])
      if data["update"] % 10 == 0:
        step_tens.append(data["update"])
        league_scores.append(data["league_average_score"])
  plt.plot(steps, raw_scores, label="raw_score_mean", marker='.') # use dot .
  plt.plot(step_tens, league_scores, label="league_average_score", marker='.')
  plt.xlabel("steps")
  plt.ylabel("scores")
  plt.title("Score Curve")
  plt.legend()
  plt.grid()
  plt.savefig(output)
  plt.show()
# save to --output

if __name__ == "__main__":
  main()
