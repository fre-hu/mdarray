/// Creates a constant-sized multidimensional array containing the arguments.
///
/// This macro is used to create an array, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array containing a given list of elements:
///
/// ```
/// use mdarray::{Array, Const, array};
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, Array::<_, (Const<_>, Const<_>)>::from([[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array from a given element and shape:
///
/// ```
/// use mdarray::{Array, Const, array};
///
/// let a = array![[1; 3]; 2];
///
/// assert_eq!(a, Array::<_, (Const<_>, Const<_>)>::from([[1; 3]; 2]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! array {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>)>::from([$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::Array::<_, ($crate::Const<_>,)>::from([$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from([[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::Array::<_, ($crate::Const<_>, $crate::Const<_>)>::from([[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        $crate::Array::<_, ($crate::Const<_>,)>::from([$elem; $i])
    );
}

/// Creates a dynamically-sized multidimensional array containing the arguments.
///
/// This macro is used to create an array, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array containing a given list of elements:
///
/// ```
/// use mdarray::{DArray, darray};
///
/// let a = darray![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, DArray::<_, 2>::from([[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array from a given element and shape by cloning the element:
///
/// ```
/// use mdarray::{DArray, darray};
///
/// let a = darray![[1; 3]; 2];
///
/// assert_eq!(a, DArray::<_, 2>::from_elem([2, 3], 1));
/// ```
///
/// In the second form, like for vectors the shape does not have to be constant.
#[macro_export]
macro_rules! darray {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DArray::<_, 6>::from([$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DArray::<_, 5>::from([$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DArray::<_, 4>::from([$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DArray::<_, 3>::from([$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::DArray::<_, 2>::from([$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::DArray::<_, 1>::from([$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::DArray::<_, 6>::from_elem([$n, $m, $l, $k, $j, $i], $elem)
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::DArray::<_, 5>::from_elem([$m, $l, $k, $j, $i], $elem)
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::DArray::<_, 4>::from_elem([$l, $k, $j, $i], $elem)
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::DArray::<_, 3>::from_elem([$k, $j, $i], $elem)
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::DArray::<_, 2>::from_elem([$j, $i], $elem)
    );
    ($elem:expr; $i:expr) => (
        $crate::DArray::<_, 1>::from_elem([$i], $elem)
    );
}

/// Creates a dynamically-sized multidimensional array view containing the arguments.
///
/// This macro is used to create an array view, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array view containing a given list of elements:
///
/// ```
/// use mdarray::{DView, dview};
///
/// let a = dview![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, DView::<_, 2>::from(&[[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array view from a given element and shape:
///
/// ```
/// use mdarray::{DView, dview};
///
/// let a = dview![[1; 3]; 2];
///
/// assert_eq!(a, DView::<_, 2>::from(&[[1; 3]; 2]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! dview {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DView::<_, 6>::from(&[$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DView::<_, 5>::from(&[$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DView::<_, 4>::from(&[$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::DView::<_, 3>::from(&[$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::DView::<_, 2>::from(&[$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::DView::<_, 1>::from(&[$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::DView::<_, 6>::from(&[[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::DView::<_, 5>::from(&[[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::DView::<_, 4>::from(&[[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::DView::<_, 3>::from(&[[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::DView::<_, 2>::from(&[[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        $crate::DView::<_, 1>::from(&[$elem; $i])
    );
}

/// Creates a dynamically-sized multidimensional array containing the arguments.
///
/// This macro is for backward compatibility, use `darray` instead.
#[macro_export]
macro_rules! tensor {
    ($($x:tt)*) => {
        $crate::darray!($($x)*)
    };
}

/// Creates a constant-sized multidimensional array view containing the arguments.
///
/// This macro is used to create an array view, similar to the `vec!` macro for vectors.
/// There are two forms of this macro:
///
/// - Create an array view containing a given list of elements:
///
/// ```
/// use mdarray::{Const, View, view};
///
/// let a = view![[1, 2, 3], [4, 5, 6]];
///
/// assert_eq!(a, View::<_, (Const<_>, Const<_>)>::from(&[[1, 2, 3], [4, 5, 6]]));
/// ```
///
/// - Create an array view from a given element and shape:
///
/// ```
/// use mdarray::{Const, View, view};
///
/// let a = view![[1; 3]; 2];
///
/// assert_eq!(a, View::<_, (Const<_>, Const<_>)>::from(&[[1; 3]; 2]));
/// ```
///
/// In the second form, the argument must be an array repeat expression with constant shape.
#[macro_export]
macro_rules! view {
    ($([$([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[$([$([$([$([$([$($x),*]),+]),+]),+]),+]),+])
    );
    ($([$([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[$([$([$([$([$($x),*]),+]),+]),+]),+])
    );
    ($([$([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[$([$([$([$($x),*]),+]),+]),+])
    );
    ($([$([$($x:expr),* $(,)?]),+ $(,)?]),+ $(,)?) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[$([$([$($x),*]),+]),+])
    );
    ($([$($x:expr),* $(,)?]),+ $(,)?) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>)>::from(&[$([$($x),*]),+])
    );
    ($($x:expr),* $(,)?) => (
        $crate::View::<_, ($crate::Const<_>,)>::from(&[$($x),*])
    );
    ([[[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr]; $n:expr) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[[[[[[$elem; $i]; $j]; $k]; $l]; $m]; $n])
    );
    ([[[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr]; $m:expr) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[[[[[$elem; $i]; $j]; $k]; $l]; $m])
    );
    ([[[$elem:expr; $i:expr]; $j:expr]; $k:expr]; $l:expr) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[[[[$elem; $i]; $j]; $k]; $l])
    );
    ([[$elem:expr; $i:expr]; $j:expr]; $k:expr) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>, $crate::Const<_>)>::from(&[[[$elem; $i]; $j]; $k])
    );
    ([$elem:expr; $i:expr]; $j:expr) => (
        $crate::View::<_, ($crate::Const<_>, $crate::Const<_>)>::from(&[[$elem; $i]; $j])
    );
    ($elem:expr; $i:expr) => (
        $crate::View::<_, ($crate::Const<_>,)>::from(&[$elem; $i])
    );
}
