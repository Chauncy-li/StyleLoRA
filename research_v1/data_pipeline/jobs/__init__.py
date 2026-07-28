"""Raw DB split and cache-generation entrypoints.

Run order:
  1. python -m research_v1.data_pipeline.jobs.build_raw_split
  2. python -m research_v1.data_pipeline.jobs.build_cache_split
  3. python -m research_v1.data_pipeline.jobs.process_cache
"""
