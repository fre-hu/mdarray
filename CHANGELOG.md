# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-02-08

- Change Insert to Resize in Axis to simplify and match what is used.
- Change default strides for StridedMapping to be consistent with DenseMapping.
- Remove unused feature extern_types.
- Merge branch 'merge-array':
  - Remove reflexive AsMut/AsRef implementations for Array.
  - Merge Array and Tensor types, and reorganize to add internal buffer types.

## [0.7.2] - 2025-12-09

- Mark most functions with inline, and remove codegen units in documentation (#17).
- Deny warnings in test script and document codegen units.
- Add Miri tests.
- Add test script and fix test name.
- Remove unnecessary inline attributes.
- Fix building with --no-default-features.
- Add inline attributes for non-generic functions and document benchmarking (#17).
- Fix FromFn index initialization for dynamic-rank shapes (#16).

## [0.7.1] - 2025-10-16

- Add swap and swap_axis methods, see #14.
- Enable mismatched lifetime syntaxes warning.
- Add no_std support based on https://github.com/kulst/mdarray commit 118017a, see #10.
- Add equality comparison methods for expressions, see #13.
- Add assume_init, uninit/uninit_in and zeros/zeros_in methods, see #12.
- Deprecate reorder and add transpose methods, see #8.

## [0.7.0] - 2025-02-23

- Add Apply and BorrowMut for Owned, and remove T: Default bound for Apply.
- Add Ord for Shape and Copy for ConstShape.
- Add Owned trait for deriving type in conversions from slices and expressions.
- Make axis indexing and iteration more consistent and complete.
- Update to edition 2024.
- Implement IntoShape for array references, see #6.
- Remove Concat and bound in Reverse, and hide types for indexing.
- Improve documentation of identity permutation, see #6.
- Update Axis trait and helper types.
- Reorganize modules and cleanup.
- Add array and to_array methods, and fix issues with view indexing and ZST layout.
- Support dynamic axis arguments and permutations, simplify Dyn and add axis indexing.
- Improve conversion functions, see issue #6.
- Add placeholder dimension in reshape methods, see #6.
- Implement row, column and diagonal methods for generic shape.
- Simplify shape and layout types for subarrays.
- Minor fixes and added more checking.
- Simplify module structure.
- Remove extern type and PhantomPinned due to lack of noalias.
- Add support for dynamic rank, see #4.
- Remove slice methods and dereferencing for IntoExpr, and update documentation.
- Rename array types, see proposal in #2.
- Switch to row-major order, and simplify layout types to Dense and Strided.
- Remove derivation of output type in eval(), to align with collect().
- Improved conversions to/from Array.
- Added design notes.

## [0.6.1] - 2024-08-31

- Added missing public items.

## [0.6.0] - 2024-08-03

- Added array methods, and some further updates.
- Added array type with inline storage.
- Fix long compile times for release build.
- Update to Rust 1.79, and avoid dependency to ATPIT.
- Simplify apply and zip_with for arrays, and readd map method.
- Introduced array shape for a list of dimensions, each having static or dynamic size.
- Remove array trait, and merge array expression and view types.
- Changed expression to a trait and some further cleanup.
- Use associated type bounds and minor cleanup.
- Return expressions instead of new arrays for operators, depends on rust-lang/rust#63063.

## [0.5.0] - 2023-12-02

- Added iteration over all but one dimension, and methods for row, column and diagonal.
- Added expressions for multidimensional iteration.
- Added permutation of dimensions.
- Create views from multiple arguments instead of a tuple, and changed ranges with negative step size.
- Refactoring and added spare_capacity_mut.
- Simplify array layout types.
- Add contains method and remove implicit borrowing.
- Remove must_use annotations and enable unused_results warning.
- Improve zero-dimensional array support and minor cleanup.
- Add missing file.
- Added macros for creating arrays and array views.
- Use fixed element order.
- Merge GridBase and SpanBase to common Array type.
- Remove generic parameters for layout.
- Make DenseSpan public.

## [0.4.0] - 2022-11-03

- Change SpanBase from ZST slice to extern type.
- Fix feature attributes for tests.
- Add must_use attributes.
- Improve interface and add debug, hashing and from_elem/from_elem_in.
- Remove dependencies to nightly features.
- Use const generics for the dimension in array view iterator and split methods.
- Remove attribute to enable GAT.
- Add span index trait, including unchecked get functions.
- Move element order into dimension trait.
- Avoid redundant format types for rank < 2, and simplify indexing.
- Fix clippy warnings.
- Move indexing into submodule.
- Simplify layout methods.
- Reorganize mapping file.
- Replace &self with self for copy types.
- Rename linear to flat format.
- Replace methods not needing self with associated constants or functions.
- Replace generic parameter with associated type.
- Update documentation.
- Add support for permissive provenance, and remove ptr_metadata feature.
- Rename split functions and add split for any axis.
- Added into_split_at and into_view for array views.
- Improve functions for flattening, reformatting and reshaping.
- Add indexing for shape and strides types, and ensure inner stride is maintained in reshape.
- Further refactoring and added more methods/operators and serde support.

## [0.3.0] - 2022-01-06

- Major refactoring including type-level constants for rank, and dense/general/strided layout.
- Add into_array/into_vec and AsMut for arrays.
- Refactor static layout and add AsRef for arrays.
- Added comparisons, conversions, debug and iterators.
- Add type for aligned memory allocation.
- Avoid separate types for subarrays, remove deref from ViewBase to slice and refactoring.
- Fix generic resize.
- Store layout inline for dense 1-dimensional views, and added missing asserts.
- Minimum alignment based on target features.

## [0.2.0] - 2021-09-11

- Renamed array types, added license files and updated version.
- Refactor and added subarrays.

## [0.1.0] - 2021-08-24

- Initial version.
