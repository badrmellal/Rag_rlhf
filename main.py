"""
Main script to run the complete RAG pipeline project.
This script orchestrates the entire process from data preparation to evaluation.
"""

import os
import argparse
from data_preparation import prepare_dataset
from build_index import build_indices
from rag_pipeline import RAGPipeline
from evaluation import evaluate
from app import create_demo
from reward_model import train_reward_model
from rlhf import RLHFTrainer


def main(args):
    """Main execution function"""
    print("Starting DarijaBridge RAG project...")

    # Step 1: Data preparation (if needed)
    if args.prepare_data or not os.path.exists('data/darija_bridge_processed.parquet'):
        print("\n--- Step 1: Preparing dataset ---")
        df, train_df, val_df, test_df = prepare_dataset(sample_size=args.sample_size)

    # Step 2: Build indices (if needed)
    if args.build_indices or not os.path.exists('indices/rag_indices.pkl'):
        print("\n--- Step 2: Building indices ---")
        indices_data = build_indices()

    # Step 3: Run evaluation (if requested)
    if args.evaluate:
        print("\n--- Step 3: Evaluating the system ---")
        metrics = evaluate(num_samples=args.eval_samples, direction=args.direction)

    if args.train_rlhf:
        print("\n--- Step 3.5: Training reward model ---")
        reward_model = train_reward_model(num_epochs=args.reward_epochs)

        print("\n--- Step 3.6: Training with RLHF ---")
        rlhf_trainer = RLHFTrainer()
        rlhf_trainer.train_with_ppo(num_iterations=args.rlhf_iterations)

    # Step 4: Launch demo (if requested)
    if args.launch_demo:
        print("\n--- Step 4: Launching demo ---")
        demo = create_demo()
        demo.launch(share=args.share_demo)

    print("\nDarijaBridge RAG project completed successfully!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DarijaBridge RAG Project")

    parser.add_argument("--prepare_data", action="store_true", help="Prepare the dataset")
    parser.add_argument("--build_indices", action="store_true", help="Build FAISS indices")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the system")
    parser.add_argument("--launch_demo", action="store_true", help="Launch the Gradio demo")

    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Number of examples to sample (default: 10000, None for all)")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples to use for evaluation")
    parser.add_argument("--direction", type=str, default="en_to_darija", choices=["en_to_darija", "darija_to_en"],
                        help="Translation direction")
    parser.add_argument("--share_demo", action="store_true", help="Share the Gradio demo publicly")

    parser.add_argument("--train_rlhf", action="store_true", help="Train with RLHF")
    parser.add_argument("--reward_epochs", type=int, default=3, help="Number of epochs for reward model")
    parser.add_argument("--rlhf_iterations", type=int, default=100, help="Number of RLHF iterations")


    args = parser.parse_args()

    # If no specific action is specified, do everything
    if not (args.prepare_data or args.build_indices or args.evaluate or args.launch_demo):
        args.prepare_data = True
        args.build_indices = True
        args.evaluate = True
        args.launch_demo = True

    main(args)