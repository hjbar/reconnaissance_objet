from typing import Any, Dict, Union
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image, ExifTags  # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
from numbers import Number
from typing import Callable, List, Optional, Iterable, Union, Any
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore


def crop_image(img: Image.Image, crop_size: int = 32) -> Image.Image:
    """Crop a PIL image to crop_size x crop_size."""
    # First determine the bounding box
    width, height = img.size
    new_width = 1.0 * crop_size
    new_height = 1.0 * crop_size
    left = round((width - new_width) / 2)
    top = round((height - new_height) / 2)
    right = round((width + new_width) / 2)
    bottom = round((height + new_height) / 2)
    # Then crop the image to that box
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def image_to_vector(img: Image.Image) -> np.ndarray:
    """Return the values of all the pixels of the Image as a vector"""
    return np.array(img).ravel()


def image_to_series(img: Image.Image) -> np.ndarray:
    """Return the values of all the pixels of the Image as a series"""
    # Note: the transparency layer is ignored.
    return pd.Series(np.array(img)[:, :, :3].ravel())


def df_cross_validate(df, sklearn_model, sklearn_metric, n=10, verbose=False):
    """Compute the performance and error bars of a classifier by cross validation

    Input:
    - a data table whose last column contains the ground truth
    - a Scikit-learn model; or anything that quacks like it
    - a Scikit-learn performance metric;
      it can either be an loss (error rate) or an accuracy (success rate).
    """
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    SSS = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=5)
    Perf_tr = np.zeros([n, 1])
    Perf_te = np.zeros([n, 1])
    k = 0
    for train_index, test_index in SSS.split(X, Y):
        # if verbose:
        #    print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        sklearn_model.fit(Xtrain, Ytrain.ravel())
        Ytrain_predicted = sklearn_model.predict(Xtrain)
        perf_tr = sklearn_metric(Ytrain.ravel(), Ytrain_predicted.ravel())
        # if verbose:
        #    print("TRAINING PERFORMANCE METRIC", perf_tr)
        Perf_tr[k] = perf_tr
        Ytest_predicted = sklearn_model.predict(Xtest)
        perf_te = sklearn_metric(Ytest.ravel(), Ytest_predicted.ravel())
        # if verbose:
        #    print("TEST PERFORMANCE METRIC:", perf_te)
        Perf_te[k] = perf_te
        k = k + 1

    perf_tr_ave = np.mean(Perf_tr)
    perf_te_ave = np.mean(Perf_te)
    sigma_tr = np.std(Perf_tr)
    sigma_te = np.std(Perf_te)
    if verbose:
        metric_name = sklearn_metric.__name__.upper()
        print(
            "*** AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f} ***".format(
                metric_name, perf_tr_ave, sigma_tr
            )
        )
        print(
            "*** AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f} ***".format(
                metric_name, perf_te_ave, sigma_te
            )
        )
    return (perf_tr_ave, sigma_tr, perf_te_ave, sigma_te)


def check_na(df):
    """Finds whether there are missing values."""
    # I used this post
    # https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null
    na_columns = df.columns[df.isna().any()]
    found_na = df[df.isna().any(axis=1)][na_columns]
    print(found_na.head())
    return found_na


def standardize_df(df):
    """Standardize all the columns except the last one (target values)."""
    df_scaled = (df - df.mean()) / df.std()
    df_scaled.iloc[:, -1] = df.iloc[:, -1]
    return df_scaled


def difference_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract a numpy array D = R-(G+B)/2 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    D = R - (G + B) / 2
    return D


def value_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract a numpy array V = (R+G+B)/3 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    V = (R + G + B) / 3
    return V


def foreground_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    M = np.array(img)  # In case this is not yet a Numpy array
    G = np.min(M[:, :, 0:3], axis=2)
    F = G < theta
    return F


def foreground_redness_filter(
    img: Union[Image.Image, np.ndarray], theta: float = 2 / 3
) -> np.ndarray:
    """Extract a numpy array with True as foreground
    and False as background from a PIL image.
    Parameter theta is a relative binarization threshold."""
    D = difference_filter(img)
    V = value_filter(img)
    F0 = np.maximum(D, V)
    threshold = theta * (np.max(F0) - np.min(F0))
    F = F0 > threshold
    return F


def invert_if_light_background(img: np.ndarray) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    if np.count_nonzero(img) > np.count_nonzero(np.invert(img)):
        return np.invert(img)
    else:
        return img


def crop_around_center(img, center, crop_size=32, verbose=False):
    """Crop a PIL image to crop_size x crop_size."""
    # First determine the center of gravity
    x0, y0 = center[1], center[0]
    # Initialize the image with the old_style cropped image
    # cropped_img = crop_image(img)
    if verbose:
        print(x0, y0)
    # Determine the bounding box
    width, height = img.size
    if verbose:
        print(width, height)
    new_width = 1.0 * crop_size
    new_height = 1.0 * crop_size
    if verbose:
        print(new_width, new_height)
    left = round(x0 - new_width / 2)
    top = round(y0 - new_height / 2)
    right = round(x0 + new_width / 2)
    bottom = round(y0 + new_height / 2)
    if verbose:
        print(left, top, right, bottom)
    # Then crop the image to that box
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def convert_image(img: Image.Image, verbose=True) -> np.ndarray:
    """Return a vector of flattened pixel values of a cropped image."""
    return np.array(img).ravel()


def transparent_background(
    img: Union[Image.Image, np.ndarray], foreground: np.ndarray
) -> Image.Image:
    """Return a copy of the image with background set to transparent.

    Which pixels belong to the background is specified by
    `foreground`, a 2D array of booleans of the same size as the
    image.
    """
    M = np.array(img)
    M[~foreground] = [0, 0, 0]
    return Image.fromarray(np.insert(M, 3, foreground * 255, axis=2))


def contours(img: Image.Image) -> np.ndarray:
    M = np.array(img)
    contour_horizontal = np.abs(M[:-1, 1:] ^ M[:-1, 0:-1])
    contour_vertical = np.abs(M[1:, :-1] ^ M[0:-1, :-1])
    return contour_horizontal + contour_vertical


def redness_filter(img: Image.Image) -> float:
    """Return the redness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    return R - G


def redness(img: Image.Image) -> float:
    """Return the redness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    F = M[:, :, 3] > 0
    return np.mean(R[F]) - np.mean(G[F])


def greenness(img: Image.Image) -> float:
    """Return the greenness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    F = M[:, :, 3] > 0
    return np.mean(G[F]) - np.mean(B[F])


def blueness(img: Image.Image) -> float:
    """Return the blueness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    B = M[:, :, 2] * 1.0
    R = M[:, :, 0] * 1.0
    F = M[:, :, 3] > 0
    return np.mean(B[F]) - np.mean(R[F])


def elongation(img: Image.Image) -> float:
    """Extract the scalar value elongation from a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    F = M[:, :, 3] > 0
    # Build the cloud of points given by the foreground image pixels
    xy = np.argwhere(F)
    if len(xy) <= 5:
        # Not enough points to compute a meaningful elongation
        return np.NaN
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    return s[0] / s[1]


def perimeter(img: Image.Image) -> float:
    """Extract the scalar value perimeter from a PIL image."""
    C = contours(img)
    return np.count_nonzero(C)


def surface(img: Image.Image) -> float:
    """Extract the scalar value surface from a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    F = M[:, :, 3] > 0
    return np.count_nonzero(F)


def get_colors(img):
    """Extract various colors from a PIL image."""
    M = np.array(img)
    F = M[:, :, 3] > 0
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    Mx = np.maximum(np.maximum(R, G), B)
    Mn = np.minimum(np.minimum(R, G), B)
    C = Mx - Mn  # Chroma
    D1 = R - (G + B) / 2
    D2 = G - B
    D3 = G - (R + B) / 2
    D4 = B - R
    D5 = B - (G + R) / 2
    D6 = R - G
    # Hue
    # H1 = np.divide(D2, C)
    # H2 = np.divide(D4, C)
    # H3 = np.divide(D6, C)
    # Luminosity
    V = (R + G + B) / 3
    # Saturation
    # S = np.divide(C, V)
    # We avoid divisions so we don't get divisions by 0
    # Now compute the color features
    if not F.any():
        return pd.Series(dtype=float)
    c = np.mean(C[F])
    return pd.Series({
        "R": np.mean(R[F]),
        "G": np.mean(G[F]),
        "B": np.mean(B[F]),
        "M": np.mean(Mx[F]),
        "m": np.mean(Mn[F]),
        "C": np.mean(C[F]),
        "R-(G+B)/2": np.mean(D1[F]),
        "G-B": np.mean(D2[F]),
        "G-(R+B)/2": np.mean(D3[F]),
        "B-R": np.mean(D4[F]),
        "B-(G+R)/2": np.mean(D5[F]),
        "R-G": np.mean(D6[F]),
        "(G-B)/C": np.mean(D2[F]) / c,
        "(B-R)/C": np.mean(D4[F]) / c,
        "(R-G)/C": np.mean(D5[F]) / c,
        "(R+G+B)/3": np.mean(V[F]),
        "C/V": c / np.mean(V[F]),
        })


def feature_learning_curve(data_df, sklearn_model, sklearn_metric):
    """Run cross-validation on nested subsets of features
    generated by the Pearson correlated coefficient."""
    print(sklearn_model.__class__.__name__.upper())
    corr = data_df.corr()
    sval = corr["class"][:-1].abs().sort_values(ascending=False)
    ranked_columns = sval.index.values
    print(ranked_columns)
    result_df = pd.DataFrame(columns=["perf_tr", "std_tr", "perf_te", "std_te"])
    for k in range(len(ranked_columns)):
        df = data_df[np.append(ranked_columns[0 : k + 1], "class")]
        rdf = pd.DataFrame(
            data=[df_cross_validate(df, sklearn_model, sklearn_metric)],
            columns=["perf_tr", "std_tr", "perf_te", "std_te"],
        )
        result_df = pd.concat([result_df, rdf], ignore_index=True)
    return result_df, ranked_columns


def systematic_model_experiment(data_df, model_name, model_list, sklearn_metric):
    """Run cross-validation on a bunch of models and collect the results."""
    result_df = pd.DataFrame(columns=["perf_tr", "std_tr", "perf_te", "std_te"])
    for name, model in zip(model_name, model_list):
        result_df.loc[name] = df_cross_validate(data_df, model, sklearn_metric)
    return result_df


def highlight_above_median(s):
    """Highlight values in a series above their median."""
    medval = s.median()
    return ["background-color: cyan" if v > medval else "" for v in s]


def analyze_model_experiments(result_df):
    tebad = result_df.perf_te < result_df.perf_te.median()
    trbad = result_df.perf_tr < result_df.perf_tr.median()
    overfitted = tebad & ~trbad
    underfitted = tebad & trbad
    result_df["Overfitted"] = overfitted
    result_df["Underfitted"] = underfitted
    return result_df.style.apply(highlight_above_median)


def extract_metadata(img: Image.Image) -> pd.Series:
    return pd.Series({ExifTags.TAGS[key]: value
                      for key, value in img.getexif().items()},
                     dtype=object)

"""
Nouvelles fonctions ou fonctions non-fournies dans le fichier originel.
"""

def grayscale(img : Image.Image):

    M = np.array(img)

    r, g, b = M[:, :, 0], M[:, :, 1], M[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


class MatchedFilter:
    ''' Matched filter class to extract a feature from a template'''
    def __init__(self, examples: Iterable[Image.Image]):
        ''' Create the template for a matched filter; use only grayscale'''
        # Compute the average of all images after conversion to grayscale
        M = np.mean([grayscale(img)
                     for img in examples], axis=0)
        # Standardize
        self.template = (M - M.mean()) / M.std()

    def show(self):
        """Show the template"""
        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot()
        ax.imshow(self.template, cmap='gray')
        return fig

    def match(self, img: Image.Image) -> Number:
        '''Extract the matched filter value for a PIL image.'''
        # Convert to grayscale and standardize
        M = grayscale(img)
        M = (M - M.mean()) / M.std()
        # Compute scalar product with the template
        # This reinforce black and white if they agree
        return np.mean(np.multiply(self.template, M))


def matched_filter(img: Image.Image, images):
    """Renvoie un poucentage de ressemble possitif si l'image ressemble à un 0 sinon renvoie un pourcentage negatif"""

    images0 = images[0:10]
    images1 = images[10:20]

    template_0 = MatchedFilter(images0)
    template_1 = MatchedFilter(images1)

    match_0 = template_0.match(img)
    match_1 = template_1.match(img)

    if match_0 > match_1:
        return match_0
    else:
        return -1 * match_1


def yellowness_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Return a grey-level image measuring the yellowness of each pixel."""
    
    M = np.array(img)

    r, g, b = M[:, :, 0], M[:, :, 1], M[:, :, 2]
    yellow = 1.0*r + 1.0*g - 1.0*b

    return yellow


def pinkness_filter(img: Image.Image) -> float:
    """Return the pinkness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    #return R - G + B
    return 0.9922*R - 0.4235*G - 0.6196*B


def greenness_filter(img: Image.Image) -> float:
    """Return the pinkness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    return -R + G - B


def orangeness_filter(img: Image.Image) -> float:
    """Return the pinkness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    return R - 0.5*G - B


def brownness_filter(img: Image.Image) -> float:
    """Return the pinkness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    return 0.3450*R - 0.1608*G - B


def whiteness_filter(img: Image.Image) -> float:
    """Return the pinkness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    return R + G + B 

"""
def foreground_filter_taux(M, n):
    newM = np.zeros(M.shape, dtype=float)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i][j] > n:
                newM[i][j] = 255
            else:
                newM[i][j] = 0
    return newM
"""

def foreground_filter_taux(M, n):
    newM = np.zeros((32, 32, 4), dtype=int)
    for i in range(32):
        for j in range(32):
            if M[i][j] > n:
                nb = 255
            else:
                nb = 0
            newM[i][j][0] = nb
            newM[i][j][1] = nb
            newM[i][j][2] = nb
            newM[i][j][3] = nb
    return newM


def elongation_matrice(tab) -> float:
    """Extract the scalar value elongation from a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = tab
    F = M[:, :, 3] > 0
    # Build the cloud of points given by the foreground image pixels
    xy = np.argwhere(F)
    if len(xy) <= 5:
        # Not enough points to compute a meaningful elongation
        return np.NaN
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    return s[0] / s[1]


def grayscale_matrice(tab):

    M = tab

    r, g, b = M[:, :, 0], M[:, :, 1], M[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


class MatchedFilter_matrice:
    ''' Matched filter class to extract a feature from a template'''
    def __init__(self, matrices):
        ''' Create the template for a matched filter; use only grayscale'''
        # Compute the average of all images after conversion to grayscale
        M = np.mean([grayscale(mat)
                     for mat in matrices], axis=0)
        # Standardize
        self.template = (M - M.mean()) / M.std()

    def show(self):
        """Show the template"""
        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot()
        ax.imshow(self.template, cmap='gray')
        return fig

    def match(self, tab) -> Number:
        '''Extract the matched filter value for a PIL image.'''
        # Convert to grayscale and standardize
        M = grayscale(tab)
        M = (M - M.mean()) / M.std()
        # Compute scalar product with the template
        # This reinforce black and white if they agree
        return np.mean(np.multiply(self.template, M))


def matched_filter_matrice(tab, matrices):
    """Renvoie un poucentage de ressemble possitif si l'image ressemble à un 0 sinon renvoie un pourcentage negatif"""

    matrices0 = matrices[0:10]
    matrices1 = matrices[10:20]

    template_0 = MatchedFilter_matrice(matrices0)
    template_1 = MatchedFilter_matrice(matrices1)

    match_0 = template_0.match(tab)
    match_1 = template_1.match(tab)

    if match_0 > match_1:
        return match_0
    else:
        return -1 * match_1


def make_scatter_plot(
    df,
    images,
    train_index=[],
    test_index=[],
    filter=None,
    predicted_labels=[],
    show_diag=False,
    axis="normal",
    feat=None,
    theta=None,
) -> Figure:
    """This scatter plot function allows us to show the images.

    predicted_labels can either be:
                    - None (queries shown as question marks)
                    - a vector of +-1 predicted values
                    - the string "GroundTruth" (to display the test images).
    Other optional arguments:
            show_diag: add diagonal dashed line if True.
            feat and theta: add horizontal or vertical line at position theta
            axis: make axes identical if 'square'."""
    fruit = np.array(["B", "A"])

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot()

    nsample, nfeat = df.shape
    if len(train_index) == 0:
        train_index = range(nsample)
    # Plot training examples
    x = df.iloc[train_index, 0]
    y = df.iloc[train_index, 1]
    f = images.iloc[train_index]
    ax.scatter(x, y, s=750, marker="o", c="w")

    for x0, y0, img in zip(x, y, f):
        ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Plot test examples
    x = df.iloc[test_index, 0]
    y = df.iloc[test_index, 1]

    if len(predicted_labels) > 0 and not (predicted_labels == "GroundTruth"):
        label = (predicted_labels + 1) / 2
        ax.scatter(x, y, s=250, marker="s", color="c")
        for x0, y0, lbl in zip(x, y, label):
            ax.text(
                x0 - 0.03,
                y0 - 0.03,
                fruit[int(lbl)],
                color="w",
                fontsize=12,
                weight="bold",
            )
    elif predicted_labels == "GroundTruth":
        f = images.iloc[test_index]
        ax.scatter(x, y, s=500, marker="s", color="c")
        for x0, y0, img in zip(x, y, f):
            ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
            ax.add_artist(ab)
    else:  # Plot UNLABELED test examples
        f = images[test_index]
        ax.scatter(x, y, s=250, marker="s", c="c")
        ax.scatter(x, y, s=100, marker="$?$", c="w")

    if axis == "square":
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(f"$x_1$ = {df.columns[0]}")
    ax.set_ylabel(f"$x_2$ = {df.columns[1]}")

    # Add line on the diagonal
    if show_diag:
        ax.plot([-3, 3], [-3, 3], "k--")

    # Add separating line along one of the axes
    if theta is not None:
        if feat == 0:  # vertical line
            ax.plot([theta, theta], [-3, 3], "k--")
        else:  # horizontal line
            ax.plot([-3, 3], [theta, theta], "k--")

    return fig