from depression_kg.pipeline import run_pipeline

if __name__ == "__main__":
    pubmed_queries = [
        '"major depressive disorder"[Title/Abstract]',
        '"depressive disorder"[Title/Abstract] AND EEG',
        '"treatment-resistant depression"[Title/Abstract]',
        'depression[Title/Abstract] AND psychotherapy[Title/Abstract]',
        'depression[Title/Abstract] AND PHQ-9[Title/Abstract]',
        '"bipolar depression"[Title/Abstract]',
        '"postpartum depression"[Title/Abstract]',
        '"chronic depression"[Title/Abstract]',
        '"recurrent depression"[Title/Abstract]',
        '"atypical depression"[Title/Abstract]',
    ]

    summary = run_pipeline(
        pubmed_queries=pubmed_queries,
        ctgov_query="major depressive disorder",
        pubmed_retmax=100,
    )

    print("Pipeline completed.")
    print(summary)
