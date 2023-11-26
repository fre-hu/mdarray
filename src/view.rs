use crate::array::{ViewArray, ViewArrayMut};
use crate::buffer::{ViewBuffer, ViewBufferMut};
use crate::dim::{Const, Dim, Shape};
use crate::index::{panic_bounds_check, Axis, Params, ViewIndex};
use crate::layout::{Dense, Layout};
use crate::mapping::{DenseMapping, Mapping};

macro_rules! impl_view {
    ($name:tt, $buffer:tt, $as_ptr:tt, $raw_mut:tt, {$($mut:tt)?}) => {
        impl<'a, T, D: Dim, L: Layout> $name<'a, T, D, L> {
            /// Converts the array view into a one-dimensional array view.
            ///
            /// # Panics
            ///
            /// Panics if the array layout is not uniformly strided.
            pub fn into_flattened(
                $($mut)? self
            ) -> $name<'a, T, Const<1>, L::Uniform> {
                let mapping = Mapping::reshape(self.mapping(), [self.len()]);

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a remapped array view.
            ///
            /// # Panics
            ///
            /// Panics if the memory layout is not compatible with the new array layout.
            pub fn into_mapping<M: Layout>($($mut)? self) -> $name<'a, T, D, M> {
                let mapping = M::Mapping::remap(self.mapping());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a reshaped array view with similar layout.
            ///
            /// # Panics
            ///
            /// Panics if the array length is changed, or the memory layout is not compatible.
            pub fn into_shape<S: Shape>(
                $($mut)? self,
                shape: S
            ) -> $name<'a, T, S::Dim, <S::Dim as Dim>::Layout<L>> {
                let mapping = Mapping::reshape(self.mapping(), shape);

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Divides the array view into two at an index along the outer dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension.
            pub fn into_split_at(
                self,
                mid: usize,
            ) -> ($name<'a, T, D, L>, $name<'a, T, D, L>) {
                assert!(D::RANK > 0, "invalid rank");

                self.into_split_dim_at(D::RANK - 1, mid)
            }

            /// Divides the array view into two at an index along the specified dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension.
            pub fn into_split_axis_at<const DIM: usize>(
                self,
                mid: usize,
            ) -> (
                $name<'a, T, D, <Const<DIM> as Axis<D>>::Split<L>>,
                $name<'a, T, D, <Const<DIM> as Axis<D>>::Split<L>>
            )
            where
                Const<DIM>: Axis<D>
            {
                self.into_mapping().into_split_dim_at(DIM, mid)
            }

            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<P: Params, I: ViewIndex<D, L, Params = P>>(
                $($mut)? self,
                index: I
            ) -> $name<'a, T, P::Dim, P::Layout>
            {
                let (offset, mapping) = I::view_index(index, self.mapping());
                let count = if mapping.is_empty() { 0 } else { offset }; // Discard offset if empty.

                unsafe { $name::new_unchecked(self.$as_ptr().offset(count), mapping) }
            }

            /// Creates an array view from a raw pointer and layout.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and a valid array view for the given layout.
            pub unsafe fn new_unchecked(ptr: *$raw_mut T, mapping: L::Mapping<D>) -> Self {
                Self { buffer: $buffer::new_unchecked(ptr, mapping) }
            }

            fn into_split_dim_at(
                $($mut)? self,
                dim: usize,
                mid: usize
            ) -> ($name<'a, T, D, L>, $name<'a, T, D, L>) {
                if mid > self.size(dim) {
                    panic_bounds_check(mid, self.size(dim));
                }

                let left_mapping = self.mapping().resize_dim(dim, mid);
                let right_mapping = self.mapping().resize_dim(dim, self.size(dim) - mid);

                // Calculate offset for the second view if non-empty.
                let count = if mid == self.size(dim) { 0 } else { self.stride(dim) * mid as isize };

                unsafe {
                    let left = $name::new_unchecked(self.$as_ptr(), left_mapping);
                    let right = $name::new_unchecked(self.$as_ptr().offset(count), right_mapping);

                    (left, right)
                }
            }
        }
    };
}

impl_view!(ViewArray, ViewBuffer, as_ptr, const, {});
impl_view!(ViewArrayMut, ViewBufferMut, as_mut_ptr, mut, {mut});

impl<'a, T> From<&'a [T]> for ViewArray<'a, T, Const<1>, Dense> {
    fn from(slice: &'a [T]) -> Self {
        unsafe { Self::new_unchecked(slice.as_ptr(), DenseMapping::new([slice.len()])) }
    }
}

impl<'a, T> From<&'a mut [T]> for ViewArrayMut<'a, T, Const<1>, Dense> {
    fn from(slice: &'a mut [T]) -> Self {
        unsafe { Self::new_unchecked(slice.as_mut_ptr(), DenseMapping::new([slice.len()])) }
    }
}

macro_rules! impl_from_array_ref {
    ($n:tt, ($($size:tt),+), $array:tt) => {
        impl<'a, T, $(const $size: usize),+> From<&'a $array>
            for ViewArray<'a, T, Const<$n>, Dense>
        {
            fn from(array: &'a $array) -> Self {
                let mapping = if [$($size),+].contains(&0) {
                    DenseMapping::new([0; $n])
                } else {
                    DenseMapping::new([$($size),+])
                };

                unsafe { Self::new_unchecked(array.as_ptr().cast(), mapping) }
            }
        }

        impl<'a, T, $(const $size: usize),+> From<&'a mut $array>
            for ViewArrayMut<'a, T, Const<$n>, Dense>
        {
            fn from(array: &'a mut $array) -> Self {
                let mapping = if [$($size),+].contains(&0) {
                    DenseMapping::new([0; $n])
                } else {
                    DenseMapping::new([$($size),+])
                };

                unsafe { Self::new_unchecked(array.as_mut_ptr().cast(), mapping) }
            }
        }
    };
}

impl_from_array_ref!(1, (X), [T; X]);
impl_from_array_ref!(2, (X, Y), [[T; X]; Y]);
impl_from_array_ref!(3, (X, Y, Z), [[[T; X]; Y]; Z]);
impl_from_array_ref!(4, (X, Y, Z, W), [[[[T; X]; Y]; Z]; W]);
impl_from_array_ref!(5, (X, Y, Z, W, U), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array_ref!(6, (X, Y, Z, W, U, V), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<'a, T> From<&'a T> for ViewArray<'a, T, Const<0>, Dense> {
    fn from(value: &'a T) -> Self {
        unsafe { Self::new_unchecked(value, DenseMapping::default()) }
    }
}

impl<'a, T> From<&'a mut T> for ViewArrayMut<'a, T, Const<0>, Dense> {
    fn from(value: &'a mut T) -> Self {
        unsafe { Self::new_unchecked(value, DenseMapping::default()) }
    }
}
