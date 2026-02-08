# Design notes

Below are some design notes, to better understand how the library is working
internally and choices that are made. It is a complement to the documentation
in `lib.rs`.

## The `Slice` type for array references

The starting point of the design is to mimic the `Vec`, `slice` and `array`
types in Rust when possible. This includes having a similar interface and
implementing the same traits etc.

One difference is that for normal arrays in Rust, a single `slice` type is
sufficient as an array reference when dereferencing from `Vec` and `array`.
However for multi-dimensional arrays, a reference is larger than `slice` and
does not fit in the two words in a fat pointer. There have been suggestions
to have custom dynamically sized types (DSTs) that could be larger, but
unfortunately this seems to be far away in the future.

It is solved by having separate view types that can contain metadata, and that
a reference is simply a pointer to the internal metadata structure `RawSlice`.
The owned array type `Array` has the same metadata structure, which makes it
possible to dereference from both the owned array and view types to a single
reference type.

The reference type is implemented as a zero sized type (ZST), to disallow any
mutation of the metadata. Otherwise one could modify the internal state of
arrays, creating undefined behavior. Internally there are type casts to the
metadata structure to access its contents.

## Fixed sized arrays

An array shape can be defined using a combination of static and/or dynamic
dimensions. Static dimensions are included in the type and do not take up space
in the metadata. This makes it possible to have fixed sized arrays without any
metadata, except for a pointer to the array elements. One can then dereference
to the `Slice` type also for fixed sized arrays that are allocated on the stack.

When there is no metadata, a reference to `Slice` points to the array elements
and not to the metadata structure. This is handled automatically depending on
the size of the metadata.

The owned `Array` type contains a buffer where the type is selected based on
the shape type. If all dimensions are constant-sized, `StaticBuffer` is used
that stores elements inline. If at least one dimension is dynamically-sized,
`DynBuffer` is used with heap allocation.

## Array view and expression types

There are two types for array views: `View` and `ViewMut`. These are created
with the methods `expr` and `expr_mut` in `Slice`, and with other methods that
give subarray views.

In addition to being arrays views, these type are also used as iterator types.
The normal iterator types cannot be used, since they do not contain information
about multiple dimensions. This is an issue for example with the `map` and `zip`
adaptors, since the result type is internal to Rust and cannot be extended.
Furthermore, iteration over multiple dimensions with the `next` method is not
efficient.

A solution is to create a separate `Expression` trait in parallel to `Iterator`.
The trait has similar methods as the iterator trait, and it is the combinators
that are important. An expression can be encapsulated in the `Iter` type to get
a regular iterator if needed.

One observation is that expressions are similar to array views, and instead of
having separate types they are merged. This both reduces complexity and avoids
unnecessary type conversions.

When iterating over an expression, the value is consumed so that one cannot
have a partially evaluated expression. It is needed to be able to merge the
expression and view types as above, and simplifies expression building.

The `Expression` trait is not implemented for the `Array` type. The reason is
that it would give the wrong behavior, so that e.g. the result from the `map`
method is an expression and not an array. One would then also expect the input
array to be consumed, but it is not useful as default.

The `Expression` trait is also not implemented for `&Slice` and `&mut Slice`.
While it could make sense and be convenient, it unfortunately deviates from
how `Iterator` and `IntoIterator` are implemented for normal array types.

## Conversion to an expression

The `IntoExpression` trait is implemented for owned arrays and array references,
similar to `IntoIterator`. It makes it possible to automatically convert to an
expression for example in function arguments.

Additionally, there is a trait `Apply` that is implemented for the same types
as `IntoExpression`. It acts as a combination of a conversion to an expression,
applying a function and evaluating the result to an array. This is useful to
implement unary and binary operators, where the result is an array if one of the
arguments is an array as described in `lib.rs`. It makes it possible to reuse
the same memory for heap allocated arrays.

## Comparison to C++ mdarray/mdspan

The design borrows a lot on the new C++ mdarray and mdspan types. These are
very well defined and gives a standard to be followed. Some deviations are made
to align with Rust naming and conventions.

Below are the larger differences to C++ mdarray/mdspan:

- There is no accessor policy for array views and references. The reason is to
  simplify and focus on the case when array elements are directly addressable.

  One use case of the accessor policy is to have custom element alignment e.g.
  to optimize for SIMD. However an alternative is to use `Simd` as element type.
  Another use case is to have scaling and/or conjugation of elements, but this
  is left for higher level libraries.

- The owned array type is parameterized by an allocator instead of a container.
  The main reason is to be able to define the `RawSlice` structure internally
  and support dereferencing to `Slice`.

- Indexing is done with `usize` and is not parameterized. This follows how
  indexing is done in Rust, and could be extended if there is a need.
