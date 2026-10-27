"""Export the bounded Phase 3.4 discovery search and replay-backed witnesses."""

import argparse
from pathlib import Path

from export_phase34 import compact

from sirfex.search.persistence import SearchStore
from sirfex.training.checkpoints import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_id")
    parser.add_argument("--store", default="artifacts/searches")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = SearchStore(args.store).load(args.search_id)
    write_json(
        args.output,
        {
            "schema_version": "phase-3.4-search-public-evidence:1",
            "search_id": args.search_id,
            "note": "Witness command arrays omitted; original full experiments remain local.",
            **compact(source),
        },
    )
