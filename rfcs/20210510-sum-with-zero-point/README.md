# Proposal for support sum with zero-point

## Motivation
The motivation to provide such support is to enable dequantize the
asymmetrically quantized sum's src1 tensor to f32 domain before performing the
sum operation. In general we want to support following operation:
`dst[:] := scale * (dst[:] - zero_point) + op(...)`

## Proposal
The proposal is to extend API by adding the possibility of using such
functionality. Support for reference solutions should also be added. For
consistency also need to implement zero points support in standalone sum
primitive. The solution should also include the benchdnn extension.

## Design

Currently, the sum operation also supports scale. Zero point support should
be designed to keep backward compatibility.

### API extension

1. Overload `append_sum(scale, dt)` function.
    - API:
        ```cpp
        // include/oneapi/dnnl/dnnl.h
        dnnl_status_t DNNL_API dnnl_post_ops_append_sum_v3(
                dnnl_post_ops_t post_ops, float scale, dnnl_data_type_t data_type,
                int32_t zero_point);

        // include/oneapi/dnnl/dnnl.hpp
        void append_sum(float scale = 1.f,
                 memory::data_type data_type = memory::data_type::undef,
                 int32_t zero_point = 0) {
            if (data_type == memory::data_type::undef)
                error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale),
                        "could not append a sum post-op");
            else
                if (zero_point == 0)
                    error::wrap_c_api(dnnl_post_ops_append_sum_v2(
                                              get(), scale,
                                              memory::convert_to_c(data_type)),
                            "could not append a sum post-op");
                else
                    error::wrap_c_api(dnnl_post_ops_append_sum_v3(
                                              get(), scale,
                                              memory::convert_to_c(data_type)),
                                              zero_point,
                            "could not append a sum post-op");
        }
        ```
    - Pros:
        - It is backwards compatible.
        - Allows to use zero point without scale (scale == 1.0).
    - Cons:
        - It is not the simplest solution
    - Example:
        ```cpp
        // Zero point is the last parameter to keep existing code working.
        attr.append_sum(scale, dt);
        attr.append_sum(scale, dt, zp);
        ```

2. Extend existing `append_sum(scale, dt)` and `dnnl_post_ops_append_sum_v2` to
support zero point.
    - API:
        ```cpp
        // include/oneapi/dnnl/dnnl.h
        dnnl_status_t DNNL_API dnnl_post_ops_append_sum_v2(
                dnnl_post_ops_t post_ops, float scale, int32_t zero_point,
                dnnl_data_type_t data_type);

        // include/oneapi/dnnl/dnnl.hpp
            void append_sum(float scale = 1.f, int32_t zero_point = 0,
                    memory::data_type data_type = memory::data_type::undef) {
                if (data_type == memory::data_type::undef)
                    error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale),
                            "could not append a sum post-op");
                else
                    error::wrap_c_api(dnnl_post_ops_append_sum_v2(
                                              get(), scale, zero_point,
                                              memory::convert_to_c(data_type)),
                            "could not append a sum post-op");
            }
        ```
    - Pros:
        - Very simple solution.
        - Universal solution.
    - Cons:
        - It is not backwards compatible for C API.
    - Example:
        ```cpp
        // Zero point is the last parameter to keep existing code working.
        attr.append_sum(scale, dt);
        ```

3. Add possibility to use `dnnl_primitive_attr_set_zero_points` with sum.
    - Pros:
        - It will allow to use mask and array of zero-point values (instead of
          one scalar value).
    - Cons:
        - More complicated solution.
        - Probably requires backward incompatibility or adding a new function.
        - Probably not needed in real topologies.
    - Example:
        ```cpp
        attr.set_zero_points(DNNL_ARG_SRC_1, mask, zero_points);
        ```

### Sum zero-point design

1. Use two parameters - `arg_zp` and `zero_point` -
  `append_sum(scale, arg_zp, zp, dt)`:
    - Pros:
        - Compatible with scale in append_sum (one scalar value).
        - Zero-point in sum is currently only needed for src (previous dst) but
          allows to add an operation for other arguments later. It is also more
          intuitive than making assumptions for arg.
    - Cons:
        - Inconsistent with `set_zero_points` function.
        - `arg_zp` may be confused with the argument for scale.

2. Use only `zero_point` parameter - `append_sum(scale, zp, dt)` (preferred):
    - Pros:
        - Compatible with scale in append_sum (one scalar value).
        - Avoid confusing `arg_zp` with the scale argument.
    - Cons:
        - Inconsistent with `set_zero_points` function.
        - It assumes that the argument is src (previous dst), which is not
          intuitive.

3. Use parameters from `set_zero_points` function - `zp_arg`, `mask`, vector of
   `zero_points` - `append_sum(scale, zp_arg, mask, zp_vec, dt)`:
    - Pros:
        - Consistent with `set_zero_points` function.
        - It allows all the possibilities of zero_point.
        - Zero-point in sum is currently only needed for src (previous dst) but
          allows to add an operation for other arguments later. It is also more
          intuitive than making assumptions for arg.
    - Cons:
        - Incompatible with scale in append_sum.
        - Probably more complex than necessary.
        - `arg_zp` may be confused with the argument for scale.

### Benchdnn extension

1. Overwrite `SUM[:SCALE[:DATA_TYPE]]` post-ops parameter by
`SUM[:SCALE[:ZEROPOINT[:DATA_TYPE]]]` instance. (preferred)
    - Pros:
        - More compatible with api function - more intuitive.
    - Cons:
        - Existing tests must be changed

2. Overwrite `SUM[:SCALE[:DATA_TYPE]]` post-ops parameter by
`SUM[:SCALE[:DATA_TYPE[:ZEROPOINT]]]` instance.
    - Pros:
        - It does not require changing existing tests
    - Cons:
        - Less compatible with api function.
