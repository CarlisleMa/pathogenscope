"""
PathogenScope: End-to-end target discovery pipeline.

Usage:
    python scripts/run_pipeline.py --fasta data/ecoli_k12/ecoli_k12_proteome.fasta

Pipeline steps:
    1. FlashPPI: Predict proteome-scale PPIs
    2. Community detection: Louvain clustering into functional modules
    3. Annotation: Fetch UniProt annotations + target scoring
    4. Agent analysis: Downstream target validation, druggability, interface analysis
"""

import argparse
import os

import pandas as pd

from pathogenscope.flashppi.predict import FlashPPIPredictor, FlashPPIConfig
from pathogenscope.community.detect import (
    build_graph,
    detect_communities,
    community_stats,
    label_communities,
)
from pathogenscope.annotation.annotate import (
    extract_accessions,
    fetch_uniprot_batch,
    score_targets,
)
from pathogenscope.agent import (
    TargetValidationAgent,
    DruggabilityAgent,
    CommunityInterpretationAgent,
    InterfaceHotspotAgent,
    DisruptionStrategyAgent,
    ReportGeneratorAgent,
)


def main():
    parser = argparse.ArgumentParser(description="PathogenScope target discovery pipeline")
    parser.add_argument("--fasta", required=True, help="Input proteome FASTA file")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Contact score threshold")
    parser.add_argument("--top_k", type=int, default=100, help="Stage 1 retrieval top-k")
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution")
    parser.add_argument("--skip_flashppi", action="store_true", help="Skip FlashPPI, use existing predictions")
    parser.add_argument("--no_annotations", action="store_true", help="Skip UniProt annotation fetch")
    parser.add_argument("--no_agents", action="store_true", help="Skip downstream agent analysis")
    parser.add_argument("--save_contact_maps", action="store_true", help="Save contact maps for interface analysis")
    parser.add_argument("--agent_top_n", type=int, default=20, help="Number of top targets for agent analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ppi_csv = os.path.join(args.output_dir, "ppi_predictions.csv")
    community_csv = os.path.join(args.output_dir, "community_stats.csv")
    targets_csv = os.path.join(args.output_dir, "target_scores.csv")

    # --- Step 1: FlashPPI ---
    contact_maps = {}
    sequences = None
    if not args.skip_flashppi:
        print("=" * 60)
        print("STEP 1: FlashPPI Proteome-Scale PPI Prediction")
        print("=" * 60)
        config = FlashPPIConfig(
            threshold=args.threshold,
            stage1_top_k=args.top_k,
        )
        predictor = FlashPPIPredictor(config)
        if args.save_contact_maps:
            predictions, contact_maps = predictor.predict_proteome(
                args.fasta, ppi_csv, save_contact_maps=True
            )
            from pathogenscope.flashppi.predict import load_fasta
            sequences = load_fasta(args.fasta, config.max_len)
        else:
            predictions = predictor.predict_proteome(args.fasta, ppi_csv)
        print(f"  -> {len(predictions)} interactions predicted\n")
    else:
        print("Skipping FlashPPI, loading existing predictions...")
        predictions = pd.read_csv(ppi_csv)

    # --- Step 2: Community Detection ---
    print("=" * 60)
    print("STEP 2: Louvain Community Detection")
    print("=" * 60)
    G = build_graph(predictions)
    partition = detect_communities(G, resolution=args.resolution)
    stats = community_stats(G, partition)
    stats.to_csv(community_csv, index=False)

    n_communities = len(set(partition.values())) if partition else 0
    print(f"  -> {n_communities} communities across {len(G)} proteins")
    print(f"  -> Saved to {community_csv}\n")

    # --- Step 3: Annotation & Target Scoring ---
    print("=" * 60)
    print("STEP 3: Annotation & Target Scoring")
    print("=" * 60)

    annotations = None
    if not args.no_annotations:
        accessions = extract_accessions(stats["protein_id"].tolist())
        print(f"  Fetching UniProt annotations for {len(accessions)} proteins...")
        annotations = fetch_uniprot_batch(accessions)
        if len(annotations) > 0:
            print(f"  -> Got annotations for {len(annotations)} proteins")

    scored = score_targets(stats, annotations)
    scored.to_csv(targets_csv, index=False)

    # --- Step 4: Agent Analysis ---
    if not args.no_agents:
        print("\n" + "=" * 60)
        print("STEP 4: Downstream Agent Analysis")
        print("=" * 60)

        agent_config = {"top_n": args.agent_top_n}

        # 4a. Target Validation
        print("\n--- Target Validation ---")
        tv_agent = TargetValidationAgent(agent_config)
        tv_result = tv_agent.run(target_scores=scored, annotations=annotations)
        print(f"  -> {tv_result.data.get('summary', {})}")

        # 4b. Druggability Assessment
        print("\n--- Druggability Assessment ---")
        drug_agent = DruggabilityAgent(agent_config)
        drug_result = drug_agent.run(target_scores=scored, annotations=annotations)
        print(f"  -> {drug_result.data.get('summary', {})}")

        # 4c. Community Interpretation
        print("\n--- Community Interpretation ---")
        comm_agent = CommunityInterpretationAgent()
        comm_result = comm_agent.run(community_stats=stats, annotations=annotations)

        # 4d. Interface Hotspot (only if contact maps available)
        hotspot_result = None
        disruption_result = None
        if contact_maps:
            print("\n--- Interface Hotspot Analysis ---")
            hs_agent = InterfaceHotspotAgent()
            hotspot_result = hs_agent.run(
                contact_maps=contact_maps, sequences=sequences, annotations=annotations
            )

            # 4e. Disruption Strategy (needs hotspot results)
            print("\n--- Disruption Strategy ---")
            ds_agent = DisruptionStrategyAgent()
            disruption_result = ds_agent.run(
                hotspot_results=hotspot_result, target_scores=scored
            )
        else:
            print("\n  Skipping interface/disruption analysis (no contact maps)")
            print("  Re-run with --save_contact_maps to enable")

        # 4f. Generate Report
        print("\n--- Generating Target Dossier ---")
        report_agent = ReportGeneratorAgent()
        report_result = report_agent.run(
            target_scores=scored,
            validation=tv_result,
            druggability=drug_result,
            community=comm_result,
            hotspot=hotspot_result,
            disruption=disruption_result,
            output_dir=args.output_dir,
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  PPI predictions:  {ppi_csv}")
    print(f"  Community stats:  {community_csv}")
    print(f"  Target scores:    {targets_csv}")
    if not args.no_agents:
        dossier = os.path.join(args.output_dir, "target_dossier.md")
        print(f"  Target dossier:   {dossier}")
    print(f"\n  Top 10 target candidates:")
    top_cols = ["protein_id", "ppi_community", "ppi_edges", "target_score"]
    available_cols = [c for c in top_cols if c in scored.columns]
    print(scored[available_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
