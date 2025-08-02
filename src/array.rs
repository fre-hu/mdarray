use core::borrow::{Borrow, BorrowMut};
use core::fmt::{Debug, Formatter, Result};
use core::hash::{Hash, Hasher};
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr;

use crate::dim::Const;
use crate::expr::{self, IntoExpr, Iter, Map, Zip};
use crate::expr::{Apply, Expression, FromExpression, IntoExpression};
use crate::index::SliceIndex;
use crate::layout::{Dense, Layout};
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;
use crate::tensor::Tensor;
use crate::traits::Owned;
use crate::view::{View, ViewMut};

/// Multidimensional array with constant-sized dimensions and inline allocation.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct Array<T, S: ConstShape>(pub S::Inner<T>);

impl<T, S: ConstShape> Array<T, S> {
    /// Creates an array from the given element.
    pub fn from_elem(elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_expr(expr::from_elem(S::default(), elem))
    }

    /// Creates an array with the results from the given function.
    pub fn from_fn<F: FnMut(&[usize]) -> T>(f: F) -> Self {
        Self::from_expr(expr::from_fn(S::default(), f))
    }

    /// Converts an array with a single element into the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the array length is not equal to one.
    pub fn into_scalar(self) -> T {
        assert!(self.len() == 1, "invalid length");

        self.into_shape::<()>().0
    }

    /// Converts the array into a reshaped array, which must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed.
    pub fn into_shape<I: ConstShape>(self) -> Array<T, I> {
        assert!(I::default().len() == self.len(), "length must not change");

        let me = ManuallyDrop::new(self);

        unsafe { mem::transmute_copy(&me) }
    }

    /// Returns an array with the same shape, and the given closure applied to each element.
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> Array<U, S> {
        self.apply(f)
    }

    /// Creates an array with uninitialized elements.
    pub fn uninit() -> Array<MaybeUninit<T>, S> {
        let array = <MaybeUninit<Self>>::uninit();

        unsafe { mem::transmute_copy(&array) }
    }

    /// Creates an array with elements set to zero.
    ///
    /// Zero elements are created using `Default::default()`.
    pub fn zeros() -> Self
    where
        T: Default,
    {
        let mut array = Self::uninit();

        array.expr_mut().for_each(|x| {
            _ = x.write(T::default());
        });

        unsafe { array.assume_init() }
    }

    fn from_expr<E: Expression<Item = T>>(expr: E) -> Self {
        struct DropGuard<'a, T, S: ConstShape> {
            array: &'a mut MaybeUninit<Array<T, S>>,
            index: usize,
        }

        impl<T, S: ConstShape> Drop for DropGuard<'_, T, S> {
            fn drop(&mut self) {
                let ptr = self.array.as_mut_ptr() as *mut T;

                unsafe {
                    ptr::slice_from_raw_parts_mut(ptr, self.index).drop_in_place();
                }
            }
        }

        // Ensure that the shape is valid.
        _ = expr.shape().with_dims(|dims| S::from_dims(dims));

        let mut array = MaybeUninit::uninit();
        let mut guard = DropGuard { array: &mut array, index: 0 };

        let ptr = guard.array.as_mut_ptr() as *mut E::Item;

        expr.for_each(|x| unsafe {
            ptr.add(guard.index).write(x);
            guard.index += 1;
        });

        mem::forget(guard);

        unsafe { array.assume_init() }
    }
}

impl<T, S: ConstShape> Array<MaybeUninit<T>, S> {
    /// Converts the array element type from `MaybeUninit<T>` to `T`.
    ///
    /// # Safety
    ///
    /// All elements in the array must be initialized, or the behavior is undefined.
    pub unsafe fn assume_init(self) -> Array<T, S> {
        unsafe { mem::transmute_copy(&self) }
    }
}

impl<'a, T, U, S: ConstShape> Apply<U> for &'a Array<T, S> {
    type Output<F: FnMut(&'a T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a T, I::Item)) -> U,
    {
        self.expr().zip(expr).map(f)
    }
}

impl<'a, T, U, S: ConstShape> Apply<U> for &'a mut Array<T, S> {
    type Output<F: FnMut(&'a mut T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a mut T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a mut T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr_mut().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a mut T, I::Item)) -> U,
    {
        self.expr_mut().zip(expr).map(f)
    }
}

impl<T, U, S: ConstShape> Apply<U> for Array<T, S> {
    type Output<F: FnMut(T) -> U> = Array<U, S>;
    type ZippedWith<I: IntoExpression, F: FnMut((T, I::Item)) -> U> = Array<U, S>;

    fn apply<F: FnMut(T) -> U>(self, f: F) -> Array<U, S> {
        Array::from_expr(self.into_expr().map(f))
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Array<U, S>
    where
        F: FnMut((T, I::Item)) -> U,
    {
        Array::from_expr(self.into_expr().zip(expr).map(f))
    }
}

impl<T, U: ?Sized, S: ConstShape> AsMut<U> for Array<T, S>
where
    Slice<T, S>: AsMut<U>,
{
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, U: ?Sized, S: ConstShape> AsRef<U> for Array<T, S>
where
    Slice<T, S>: AsRef<U>,
{
    fn as_ref(&self) -> &U {
        (**self).as_ref()
    }
}

macro_rules! impl_as_mut_ref {
    (($($xyz:tt),+), $array:tt) => {
        impl<T, $(const $xyz: usize),+> AsMut<Array<T, ($(Const<$xyz>,)+)>> for $array {
            fn as_mut(&mut self) -> &mut Array<T, ($(Const<$xyz>,)+)> {
                unsafe { &mut *(self as *mut Self as *mut Array<T, ($(Const<$xyz>,)+)>) }
            }
        }

        impl<T, $(const $xyz: usize),+> AsRef<Array<T, ($(Const<$xyz>,)+)>> for $array {
            fn as_ref(&self) -> &Array<T, ($(Const<$xyz>,)+)> {
                unsafe { &*(self as *const Self as *const Array<T, ($(Const<$xyz>,)+)>) }
            }
        }
    };
}

impl_as_mut_ref!((X), [T; X]);
impl_as_mut_ref!((X, Y), [[T; Y]; X]);
impl_as_mut_ref!((X, Y, Z), [[[T; Z]; Y]; X]);
impl_as_mut_ref!((X, Y, Z, W), [[[[T; W]; Z]; Y]; X]);
impl_as_mut_ref!((X, Y, Z, W, U), [[[[[T; U]; W]; Z]; Y]; X]);
impl_as_mut_ref!((X, Y, Z, W, U, V), [[[[[[T; V]; U]; W]; Z]; Y]; X]);

impl<T, S: ConstShape> Borrow<Slice<T, S>> for Array<T, S> {
    fn borrow(&self) -> &Slice<T, S> {
        self
    }
}

impl<T, S: ConstShape> BorrowMut<Slice<T, S>> for Array<T, S> {
    fn borrow_mut(&mut self) -> &mut Slice<T, S> {
        self
    }
}

impl<T: Debug, S: ConstShape> Debug for Array<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        (**self).fmt(f)
    }
}

impl<T, S: ConstShape> Deref for Array<T, S> {
    type Target = Slice<T, S>;

    fn deref(&self) -> &Self::Target {
        _ = S::default().checked_len().expect("invalid length");

        unsafe { &*(self as *const Self as *const Slice<T, S>) }
    }
}

impl<T, S: ConstShape> DerefMut for Array<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        _ = S::default().checked_len().expect("invalid length");

        unsafe { &mut *(self as *mut Self as *mut Slice<T, S>) }
    }
}

impl<T, S: ConstShape> From<Tensor<T, S>> for Array<T, S> {
    fn from(value: Tensor<T, S>) -> Self {
        Self::from_expr(value.into_expr())
    }
}

impl<'a, T: 'a + Clone, S: ConstShape, L: Layout, I> From<I> for Array<T, S>
where
    I: IntoExpression<IntoExpr = View<'a, T, S, L>>,
{
    fn from(value: I) -> Self {
        Self::from_expr(value.into_expr().cloned())
    }
}

macro_rules! impl_from_array {
    (($($xyz:tt),+), $array:tt) => {
        impl<T: Clone $(,const $xyz: usize)+> From<&$array> for Array<T, ($(Const<$xyz>,)+)> {
            fn from(array: &$array) -> Self {
                Self(array.clone())
            }
        }

        impl<T $(,const $xyz: usize)+> From<Array<T, ($(Const<$xyz>,)+)>> for $array {
            fn from(array: Array<T, ($(Const<$xyz>,)+)>) -> Self {
                array.0
            }
        }

        impl<T $(,const $xyz: usize)+> From<$array> for Array<T, ($(Const<$xyz>,)+)> {
            fn from(array: $array) -> Self {
                Self(array)
            }
        }
    };
}

impl_from_array!((X), [T; X]);
impl_from_array!((X, Y), [[T; Y]; X]);
impl_from_array!((X, Y, Z), [[[T; Z]; Y]; X]);
impl_from_array!((X, Y, Z, W), [[[[T; W]; Z]; Y]; X]);
impl_from_array!((X, Y, Z, W, U), [[[[[T; U]; W]; Z]; Y]; X]);
impl_from_array!((X, Y, Z, W, U, V), [[[[[[T; V]; U]; W]; Z]; Y]; X]);

impl<T, S: ConstShape> FromExpression<T, S> for Array<T, S> {
    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::from_expr(expr.into_expr())
    }
}

impl<T: Hash, S: ConstShape> Hash for Array<T, S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, S: ConstShape, I: SliceIndex<T, S, Dense>> Index<I> for Array<T, S> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: ConstShape, I: SliceIndex<T, S, Dense>> IndexMut<I> for Array<T, S> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: ConstShape> IntoExpression for &'a Array<T, S> {
    type Shape = S;
    type IntoExpr = View<'a, T, S>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: ConstShape> IntoExpression for &'a mut Array<T, S> {
    type Shape = S;
    type IntoExpr = ViewMut<'a, T, S>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<T, S: ConstShape> IntoExpression for Array<T, S> {
    type Shape = S;
    type IntoExpr = IntoExpr<Array<ManuallyDrop<T>, S>>;

    fn into_expr(self) -> Self::IntoExpr {
        _ = S::default().checked_len().expect("invalid length");

        let me = ManuallyDrop::new(self);

        unsafe { IntoExpr::new(mem::transmute_copy(&me)) }
    }
}

impl<'a, T, S: ConstShape> IntoIterator for &'a Array<T, S> {
    type Item = &'a T;
    type IntoIter = Iter<View<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: ConstShape> IntoIterator for &'a mut Array<T, S> {
    type Item = &'a mut T;
    type IntoIter = Iter<ViewMut<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, S: ConstShape> IntoIterator for Array<T, S> {
    type Item = T;
    type IntoIter = Iter<IntoExpr<Array<ManuallyDrop<T>, S>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_expr().into_iter()
    }
}

impl<T, S: ConstShape> Owned<T, S> for Array<T, S> {
    type WithConst<const N: usize> = S::WithConst<T, N, Self>;

    fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone,
    {
        self.assign(slice);
    }
}
