import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import calculate_expected_series_win
from config import MC_ITERATIONS, MC_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description="Full CS2 Series Win Predictor (Veto Sim + Neural Network)")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    parser.add_argument("--format", choices=["bo1", "bo3", "bo5"], default="bo3", help="Series format (default: bo3)")
    parser.add_argument("--starts-veto", help="Which team starts the veto (team_a/a or team_b/b). Defaults to random.")
    parser.add_argument("--iters", type=int, default=MC_ITERATIONS, help=f"Monte Carlo iterations for veto simulation (default: {MC_ITERATIONS})")
    parser.add_argument("--threshold", type=float, default=MC_THRESHOLD, help=f"Probability truncation threshold (default: {MC_THRESHOLD})")
    
    args = parser.parse_args()

    # ASCII decoration
    print("\n" + "="*60)
    print(f" EXPECTED SERIES WIN PREDICTION: {args.team_a} vs {args.team_b}")
    print(f" Format: {args.format.upper()} | MC Iters: {args.iters:,} | Threshold: {args.threshold*100:.0f}%")
    if args.starts_veto:
        print(f" Starting Veto Team: {args.starts_veto}")
    else:
        print(f" Starting Veto Team: 50/50 Randomized")
    print("="*60)

    try:
        # Run the integrated calculation
        results = calculate_expected_series_win(
            args.team_a, 
            args.team_b, 
            series_format=args.format, 
            threshold=args.threshold, 
            iters=args.iters, 
            starts_veto=args.starts_veto
        )
        
        prob = results["expected_win_prob"]
        t_a_id = results["team_a_id"]
        t_b_id = results["team_b_id"]
        ctx = results["predictor_ctx"]
        seq_counts = results["sequence_counts"]

        # Display Dataset Stats
        print(f"\n DATASET STATISTICS:")
        n_a = len(ctx.gen_histories.get(t_a_id, []))
        n_b = len(ctx.gen_histories.get(t_b_id, []))
        print(f" {t_a_id:20} | {n_a:4} maps in database")
        print(f" {t_b_id:20} | {n_b:4} maps in database")
        print("-" * 60)

        # Display Top 5 Sequences
        print(f"\n MOST LIKELY VETO SEQUENCES (TOP 5):")
        sorted_seqs = sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (seq, count) in enumerate(sorted_seqs):
            s_prob = (count / args.iters) * 100
            print(f" {i+1}. {s_prob:5.1f}% | {seq}")
        print("-" * 60)
        
        print(f"\n FINAL EXPECTED SERIES PROBABILITIES:")
        print(f" {t_a_id:20} | {prob*100:6.2f}%")
        print(f" {t_b_id:20} | {(1-prob)*100:6.2f}%")
        print("-" * 60)
        print(f" (Logic: Weighted sum of all veto paths with cumulative probability >= {args.threshold*100:.0f}%)")
        print("="*60 + "\n")

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
