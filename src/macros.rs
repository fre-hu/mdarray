/// Creates an inline multidimensional array containing the arguments.
///
/// This macro is used to create an array, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array containing a given list of elements:
///
/// ```
/// use mdarray::{array, Array};
///
/// let a = array![[1, 2], [3, 4]];
///
/// assert_eq!(a, Array::from([[1, 2], [3, 4]]));
/// ```
///
/// - Create an array from a given element and shape:
///
/// ```
/// use mdarray::{array, Const, Array};
///
/// let a = array![[1; 2]; 3];
///
/// assert_eq!(a, Array::from([[1; 2]; 3]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! array {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, (_, _, _, _, _, _)>::from([$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, (_, _, _, _, _)>::from([$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, (_, _, _, _)>::from([$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, (_, _, _)>::from([$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::Array::<_, (_, _)>::from([$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::Array::<_, _>::from([$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::Array::<_, (_, _, _, _, _, _)>::from([[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::Array::<_, (_, _, _, _, _)>::from([[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::Array::<_, (_, _, _, _)>::from([[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::Array::<_, (_, _, _)>::from([[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::Array::<_, (_, _)>::from([[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        $crate::Array::<_, _>::from([$elem; $i])
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
/// use mdarray::{expr, expr::Expr};
///
/// let a = expr![[1, 2], [3, 4]];
///
/// assert_eq!(a, Expr::from(&[[1, 2], [3, 4]]));
/// ```
///
/// - Create an array view from a given element and shape:
///
/// ```
/// use mdarray::{expr, expr::Expr};
///
/// let a = expr![[1; 2]; 3];
///
/// assert_eq!(a, Expr::from(&[[1; 2]; 3]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! expr {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::expr::Expr::<_, $crate::Rank<6>>::from(&[$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::expr::Expr::<_, $crate::Rank<5>>::from(&[$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::expr::Expr::<_, $crate::Rank<4>>::from(&[$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::expr::Expr::<_, $crate::Rank<3>>::from(&[$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::expr::Expr::<_, $crate::Rank<2>>::from(&[$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::expr::Expr::<_, $crate::Rank<1>>::from(&[$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::expr::Expr::<_, $crate::Rank<6>>::from(&[[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::expr::Expr::<_, $crate::Rank<5>>::from(&[[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::expr::Expr::<_, $crate::Rank<4>>::from(&[[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::expr::Expr::<_, $crate::Rank<3>>::from(&[[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::expr::Expr::<_, $crate::Rank<2>>::from(&[[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        $crate::expr::Expr::<_, $crate::Rank<1>>::from(&[$elem; $i])
    );
}

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
/// assert_eq!(a, Grid::from_elem([2, 3], 1));
/// ```
///
/// In the second form, like for vectors the shape does not have to be constant.
#[macro_export]
macro_rules! grid {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DGrid::<_, 6>::from([$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DGrid::<_, 5>::from([$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DGrid::<_, 4>::from([$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DGrid::<_, 3>::from([$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::DGrid::<_, 2>::from([$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::DGrid::<_, 1>::from([$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::DGrid::<_, 6>::from_elem([$i, $j, $k, $l, $m, $n], $elem)
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::DGrid::<_, 5>::from_elem([$i, $j, $k, $l, $m], $elem)
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::DGrid::<_, 4>::from_elem([$i, $j, $k, $l], $elem)
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::DGrid::<_, 3>::from_elem([$i, $j, $k], $elem)
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::DGrid::<_, 2>::from_elem([$i, $j], $elem)
    );
    ($elem:expr; $i:expr) => (
        $crate::DGrid::<_, 1>::from_elem([$i], $elem)
    );
}
