/// Creates an inline multidimensional array containing the arguments.
///
/// This macro is used to create an array, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array containing a given list of elements:
///
/// ```
/// use mdarray::{Array, array};
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, Array::from([[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array from a given element and shape:
///
/// ```
/// use mdarray::{array, Const, Array};
///
/// let a = array![[1; 3]; 2];
///
/// assert_eq!(a, Array::from([[1; 3]; 2]));
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
        $crate::Array::<_, (_,)>::from([$($x),*])
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
        $crate::Array::<_, (_,)>::from([$elem; $i])
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
/// use mdarray::{DTensor, tensor};
///
/// let a = tensor![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, DTensor::<_, 2>::from([[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array from a given element and shape by cloning the element:
///
/// ```
/// use mdarray::{tensor, Tensor};
///
/// let a = tensor![[1; 3]; 2];
///
/// assert_eq!(a, Tensor::from_elem([2, 3], 1));
/// ```
///
/// In the second form, like for vectors the shape does not have to be constant.
#[macro_export]
macro_rules! tensor {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DTensor::<_, 6>::from([$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DTensor::<_, 5>::from([$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DTensor::<_, 4>::from([$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DTensor::<_, 3>::from([$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::DTensor::<_, 2>::from([$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::DTensor::<_, 1>::from([$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::DTensor::<_, 6>::from_elem([$n, $m, $l, $k, $j, $i], $elem)
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::DTensor::<_, 5>::from_elem([$m, $l, $k, $j, $i], $elem)
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::DTensor::<_, 4>::from_elem([$l, $k, $j, $i], $elem)
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::DTensor::<_, 3>::from_elem([$k, $j, $i], $elem)
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::DTensor::<_, 2>::from_elem([$j, $i], $elem)
    );
    ($elem:expr; $i:expr) => (
        $crate::DTensor::<_, 1>::from_elem([$i], $elem)
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
/// use mdarray::{DView, view};
///
/// let a = view![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, DView::<_, 2>::from(&[[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array view from a given element and shape:
///
/// ```
/// use mdarray::{view, DView};
///
/// let a = view![[1; 3]; 2];
///
/// assert_eq!(a, DView::<_, 2>::from(&[[1; 3]; 2]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! view {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, $crate::Rank<6>>::from(&[$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, $crate::Rank<5>>::from(&[$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, $crate::Rank<4>>::from(&[$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, $crate::Rank<3>>::from(&[$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::View::<_, $crate::Rank<2>>::from(&[$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::View::<_, $crate::Rank<1>>::from(&[$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::View::<_, $crate::Rank<6>>::from(&[[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::View::<_, $crate::Rank<5>>::from(&[[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::View::<_, $crate::Rank<4>>::from(&[[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::View::<_, $crate::Rank<3>>::from(&[[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::View::<_, $crate::Rank<2>>::from(&[[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        $crate::View::<_, $crate::Rank<1>>::from(&[$elem; $i])
    );
}
