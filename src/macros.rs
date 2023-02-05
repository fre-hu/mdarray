/// Creates a dense multidimensional array containing the arguments.
///
/// This macro is used to create an array, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array containing a given list of elements:
///
/// ```
/// use mdarray::{grid, Grid};
///
/// let a = grid![[1, 2], [3, 4]];
///
/// assert_eq!(a, Grid::from([[1, 2], [3, 4]]));
/// ```
///
/// - Create an array from a given element and shape by cloning the element:
///
/// ```
/// use mdarray::{grid, Grid};
///
/// let a = grid![[1; 2]; 3];
///
/// assert_eq!(a, Grid::from_elem([2, 3], &1));
/// ```
///
/// In the second form, like for vectors the shape does not have to be constant.
#[macro_export]
macro_rules! grid {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        Grid::<_, 6>::from([$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        Grid::<_, 5>::from([$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        Grid::<_, 4>::from([$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        Grid::<_, 3>::from([$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        Grid::<_, 2>::from([$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        Grid::<_, 1>::from([$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        Grid::<_, 6>::from_elem([$i, $j, $k, $l, $m, $n], &$elem)
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        Grid::<_, 5>::from_elem([$i, $j, $k, $l, $m], &$elem)
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        Grid::<_, 4>::from_elem([$i, $j, $k, $l], &$elem)
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        Grid::<_, 3>::from_elem([$i, $j, $k], &$elem)
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        Grid::<_, 2>::from_elem([$i, $j], &$elem)
    );
    ($elem:expr; $i:expr) => (
        Grid::<_, 1>::from_elem([$i], &$elem)
    );
}

/// Creates a multidimensional array view containing the arguments.
///
/// This macro is used to create an array view, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array view containing a given list of elements:
///
/// ```
/// use mdarray::{view, View};
///
/// let a = view![[1, 2], [3, 4]];
///
/// assert_eq!(a, View::from(&[[1, 2], [3, 4]]));
/// ```
///
/// - Create an array view from a given element and shape:
///
/// ```
/// use mdarray::{view, View};
///
/// let a = view![[1; 2]; 3];
///
/// assert_eq!(a, View::from(&[[1; 2]; 3]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! view {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        View::<_, 6>::from(&[$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        View::<_, 5>::from(&[$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        View::<_, 4>::from(&[$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        View::<_, 3>::from(&[$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        View::<_, 2>::from(&[$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        View::<_, 1>::from(&[$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        View::<_, 6>::from(&[[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        View::<_, 5>::from(&[[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        View::<_, 4>::from(&[[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        View::<_, 3>::from(&[[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        View::<_, 2>::from(&[[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        View::<_, 1>::from(&[$elem; $i])
    );
}
