"""Central configuration for the data-reduction experiments."""

CONFIG = {
    "dataset_path": r"data/photos/photos.csv",
    "queries_path": r"data/photos/queries.csv",
    "seed": 42,
    "quiet": False,

    "save_results": True,
    "results_dir": "results",  

    "run": {
        "a": True,
        "b": True,
        "c": True,
        "d": True,
    },

    # --- candidate universe D ---
    # "first"         -> first pool_size ids in the dataset
    # "random"        -> random sample of pool_size ids
    # "ids"           -> explicit ids listed below
    # "queried"       -> random sample from ids appearing in queries
    # "query_union"   -> union of ids from random queries
    # "top_frequency" -> most frequent ids in queries
    "pool": {
        "strategy": "top_frequency",
        "size": 18,
        "ids": [
            30618, 13589, 12473, 26916, 37771,
            11696, 5257, 33523, 2201, 5519,
            5342, 23778, 3975, 11802, 17505,
        ],
        "keep_size": 7,
    },

    "full": {
        "keep_size": 416
    },


    # --- methods ---
    "methods": {
        "a": {
            "mode": "pool",
        },
        "b": {
            "mode": "pool",
        },
        "c": {
            "mode": "pool",
        },
        "d": {
            "mode": "pool",
            "alpha":  1,
            "expansion": 2 # default is 2
        }
    },
}