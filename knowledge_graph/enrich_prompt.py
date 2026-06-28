from __future__ import annotations

from depression_kg.prompt_enricher import PromptEnricher, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    prompt_text = (args.text or "").strip()
    if not prompt_text:
        prompt_text = input("Enter prompt text: ").strip()

    if not prompt_text:
        raise SystemExit("No prompt text provided.")

    enricher = PromptEnricher(entities_csv=args.entities_csv, relations_csv=args.relations_csv)
    enriched_prompt = enricher.enrich_prompt(prompt_text, top_k_relations=args.top_k_relations)

    print("\n=== Enriched Prompt ===\n")
    print(enriched_prompt)


if __name__ == "__main__":
    main()

