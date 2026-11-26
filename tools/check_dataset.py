import h5py
import numpy as np

path = (
    "data/processed/"
    "fno_6cond_element_512_direct_ewma.h5"
)

conditions = [
    "25",
    "62.5",
    "75",
    "87.5",
    "full",
    "half",
]

with h5py.File(path, "r") as f:
    X = f["X_all"]
    mask = f["mask"][:] > 0

    reference_index = int(f["idx_25"][0])
    reference_q = X[reference_index, 4][mask]

    print(
        "heat_source_mode:",
        f.attrs.get("heat_source_mode"),
    )

    print(
        "Q_input_definition:",
        f.attrs.get("Q_input_definition"),
    )

    for condition in conditions:
        index = int(
            f[f"idx_{condition}"][0]
        )

        q_active = X[index, 4][mask]

        maximum_difference = float(
            np.max(
                np.abs(
                    q_active - reference_q
                )
            )
        )

        print(
            f"{condition:>6}: "
            f"mean={q_active.mean():.8f}, "
            f"max_diff_from_25="
            f"{maximum_difference:.8e}"
        )