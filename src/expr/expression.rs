#[cfg(feature = "nightly")]
use std::alloc::Allocator;

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::expr::adapters::{Cloned, Copied, Enumerate, Map, Zip};
use crate::expr::iter::Iter;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::traits::IntoCloned;

/// Trait for applying a closure and returning an existing array or an expression.
pub trait Apply<T>: IntoExpression {
    /// The resulting type after applying a closure.
    type Output<F: FnMut(Self::Item) -> T>: IntoExpression<Item = T, Shape = Self::Shape>;

    /// The resulting type after zipping elements and applying a closure.
    type ZippedWith<I: IntoExpression, F>: IntoExpression<Item = T>
    where
        F: FnMut((Self::Item, I::Item)) -> T;

    /// Returns the array or an expression with the given closure applied to each element.
    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output<F>;

    /// Returns the array or an expression with the given closure applied to zipped element pairs.
    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((Self::Item, I::Item)) -> T;
}

/// Expression trait, for multidimensional iteration.
pub trait Expression: IntoIterator {
    /// Array shape type.
    type Shape: Shape;

    /// True if the expression can be restarted from the beginning after the last element.
    const IS_REPEATABLE: bool;

    /// Returns the array shape.
    fn shape(&self) -> &Self::Shape;

    /// Creates an expression which clones all of its elements.
    fn cloned<'a, T: 'a + Clone>(self) -> Cloned<Self>
    where
        Self: Expression<Item = &'a T> + Sized,
    {
        Cloned::new(self)
    }

    /// Creates an expression which copies all of its elements.
    fn copied<'a, T: 'a + Copy>(self) -> Copied<Self>
    where
        Self: Expression<Item = &'a T> + Sized,
    {
        Copied::new(self)
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(&self, index: usize) -> usize {
        self.shape().dim(index)
    }

    /// Creates an expression which gives tuples of the current count and the element.
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
    {
        Enumerate::new(self)
    }

    /// Evaluates the expression into a new array.
    ///
    /// The resulting type is `Array` if the shape has constant-sized dimensions, or
    /// otherwise `Tensor`. If the shape type is generic, `FromExpression::from_expr`
    /// can be used to evaluate the expression into a specific array type.
    fn eval(self) -> <Self::Shape as Shape>::Owned<Self::Item>
    where
        Self: Sized,
    {
        FromExpression::from_expr(self)
    }

    /// Evaluates the expression with broadcasting and appends to the given array
    /// along the first dimension.
    ///
    /// If the array is empty, it is reshaped to match the shape of the expression.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, if the rank is not the same and
    /// at least 1, or if the first dimension is not dynamically-sized.
    fn eval_into<S: Shape, A: Allocator>(
        self,
        tensor: &mut Tensor<Self::Item, S, A>,
    ) -> &mut Tensor<Self::Item, S, A>
    where
        Self: Sized,
    {
        tensor.expand(self);
        tensor
    }

    /// Folds all elements into an accumulator by applying an operation, and returns the result.
    fn fold<T, F: FnMut(T, Self::Item) -> T>(self, init: T, f: F) -> T
    where
        Self: Sized,
    {
        Iter::new(self).fold(init, f)
    }

    /// Calls a closure on each element of the expression.
    fn for_each<F: FnMut(Self::Item)>(self, mut f: F)
    where
        Self: Sized,
    {
        self.fold((), |(), x| f(x));
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(&self) -> bool {
        self.shape().is_empty()
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.shape().len()
    }

    /// Creates an expression that calls a closure on each element.
    fn map<T, F: FnMut(Self::Item) -> T>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
    {
        Map::new(self, f)
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Creates an expression that gives tuples `(x, y)` of the elements from each expression.
    ///
    /// # Panics
    ///
    /// Panics if the expressions cannot be broadcast to a common shape.
    fn zip<I: IntoExpression>(self, other: I) -> Zip<Self, I::IntoExpr>
    where
        Self: Sized,
    {
        Zip::new(self, other.into_expr())
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item;

    #[doc(hidden)]
    fn inner_rank(&self) -> usize;

    #[doc(hidden)]
    unsafe fn reset_dim(&mut self, index: usize, count: usize);

    #[doc(hidden)]
    unsafe fn step_dim(&mut self, index: usize);

    #[cfg(not(feature = "nightly"))]
    #[doc(hidden)]
    fn clone_into_vec<T>(self, vec: &mut Vec<T>)
    where
        Self: Expression<Item: IntoCloned<T>> + Sized,
    {
        assert!(self.len() <= vec.capacity() - vec.len(), "length exceeds capacity");

        self.for_each(|x| unsafe {
            vec.as_mut_ptr().add(vec.len()).write(x.into_cloned());
            vec.set_len(vec.len() + 1);
        });
    }

    #[cfg(feature = "nightly")]
    #[doc(hidden)]
    fn clone_into_vec<T, A: Allocator>(self, vec: &mut Vec<T, A>)
    where
        Self: Expression<Item: IntoCloned<T>> + Sized,
    {
        assert!(self.len() <= vec.capacity() - vec.len(), "length exceeds capacity");

        self.for_each(|x| unsafe {
            vec.as_mut_ptr().add(vec.len()).write(x.into_cloned());
            vec.set_len(vec.len() + 1);
        });
    }
}

/// Conversion trait from an expression.
pub trait FromExpression<T, S: Shape>: Sized {
    /// Creates an array from an expression.
    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self;
}

/// Conversion trait into an expression.
pub trait IntoExpression: IntoIterator {
    /// Array shape type.
    type Shape: Shape;

    /// Which kind of expression are we turning this into?
    type IntoExpr: Expression<Item = Self::Item, Shape = Self::Shape>;

    /// Creates an expression from a value.
    fn into_expr(self) -> Self::IntoExpr;
}

impl<T, E: Expression> Apply<T> for E {
    type Output<F: FnMut(Self::Item) -> T> = Map<E, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((Self::Item, I::Item)) -> T> =
        Map<Zip<Self, I::IntoExpr>, F>;

    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output<F> {
        self.map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((Self::Item, I::Item)) -> T,
    {
        self.zip(expr).map(f)
    }
}

impl<E: Expression> IntoExpression for E {
    type Shape = E::Shape;
    type IntoExpr = E;

    fn into_expr(self) -> Self {
        self
    }
}
