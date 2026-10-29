"""Read and check the input files of the mean-variance plugin.

The files are written by ZEN-Europe into the dataset folder::

    <dataset>/mean_variance/<source>/relative_sd.csv
        Columns ``<element>, relative_sd``, e.g. ``technology, relative_sd``.
        ``relative_sd`` is the standard deviation as a share of the cost
        parameter, e.g. 0.1 = 10 % of the specific capex.
    <dataset>/mean_variance/<source>/correlation.csv
        Square matrix. Row and column labels are element names; empty cells
        are read as 0.

Every element of the model must appear in both files. Elements in the files
that are not part of the model are ignored.
"""

from pathlib import Path

import numpy as np
import pandas as pd

#: Tolerance when checking that the correlation matrix has ones on its diagonal.
DIAGONAL_TOLERANCE = 1e-4
#: Smallest eigenvalue accepted when checking that a matrix is positive
#: semi-definite (PSD). A correlation matrix must be PSD, otherwise the
#: variance can become negative and the quadratic objective is not convex.
PSD_TOLERANCE = -1e-8


def get_input_folder(dataset_path: str | Path, source_name: str) -> Path:
    """Return the folder that holds the input files of one variance source."""
    return Path(dataset_path) / "mean_variance" / source_name


def read_relative_sd(
    folder: Path, element_column: str, model_elements: list[str]
) -> pd.Series:
    """Read the relative standard deviation of every model element.

    Args:
        folder: Folder containing ``relative_sd.csv``.
        element_column: Name of the column with the element names, e.g.
            ``"technology"`` or ``"carrier"``.
        model_elements: Elements of the model that need a value.

    Returns:
        Relative standard deviation indexed by element, in the order of
        ``model_elements``.
    """
    path = folder / "relative_sd.csv"
    # "utf-8-sig" also reads files saved by Excel, which start with a BOM
    table = pd.read_csv(path, encoding="utf-8-sig")
    relative_sd = table.set_index(element_column)["relative_sd"]

    _raise_if_elements_missing(model_elements, relative_sd.index, path)
    relative_sd = relative_sd.loc[model_elements].astype(float)

    if (relative_sd < 0).any():
        negative = list(relative_sd.index[relative_sd < 0])
        raise ValueError(f"{path} has negative relative_sd values for {negative}.")
    return relative_sd


def read_correlation_matrix(
    folder: Path, model_elements: list[str], regularization: float
) -> pd.DataFrame:
    """Read the correlation matrix of the model elements and check it.

    Rows and columns are matched by name, so their order in the file does not
    matter. To make sure the matrix is numerically positive definite, it is
    shrunk slightly towards the identity matrix::

        R_used = (1 - regularization) * R + regularization * I

    Args:
        folder: Folder containing ``correlation.csv``.
        model_elements: Elements of the model; the returned matrix has exactly
            these rows and columns.
        regularization: Shrinkage towards the identity matrix (0 = none).

    Returns:
        Square correlation matrix with rows and columns ``model_elements``.
    """
    path = folder / "correlation.csv"
    matrix = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    _raise_if_elements_missing(model_elements, matrix.index, path, "rows")
    _raise_if_elements_missing(model_elements, matrix.columns, path, "columns")
    matrix = matrix.loc[model_elements, model_elements].fillna(0.0).astype(float)

    if not np.allclose(matrix.values, matrix.values.T):
        raise ValueError(f"The correlation matrix in {path} is not symmetric.")
    diagonal = np.diag(matrix.values)
    if not np.allclose(diagonal, 1.0, atol=DIAGONAL_TOLERANCE):
        wrong = list(matrix.index[~np.isclose(diagonal, 1.0, atol=DIAGONAL_TOLERANCE)])
        raise ValueError(
            f"The correlation matrix in {path} must be 1 on the diagonal, "
            f"which is not the case for {wrong}."
        )

    identity = np.eye(len(model_elements))
    matrix.loc[:, :] = (1 - regularization) * matrix.values + regularization * identity

    smallest_eigenvalue = np.linalg.eigvalsh(matrix.values).min() if len(matrix) else 0
    if smallest_eigenvalue < PSD_TOLERANCE:
        raise ValueError(
            f"The correlation matrix in {path} is not positive semi-definite "
            f"(smallest eigenvalue {smallest_eigenvalue:.2e})."
        )
    return matrix


def _raise_if_elements_missing(
    model_elements: list[str], file_elements: pd.Index, path: Path, where: str = ""
) -> None:
    """Raise a readable error if model elements are missing in an input file."""
    missing = [element for element in model_elements if element not in file_elements]
    if missing:
        location = f" ({where})" if where else ""
        raise ValueError(
            f"{path}{location} has no entry for the model elements {missing}. "
            "Add them to the file (in ZEN-Europe: data/raw_data/05-mean_variance)."
        )
