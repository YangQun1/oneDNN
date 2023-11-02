## Open Questions

There are some open questions that not included in the above proposals, we want
to discuss them here:

### Cache management policy control
Cache management policy is used to define the cache behaviors when the cache
size reaches the capacity.

We mentioned a simple policy above: not cache the new coming tensors and not
evict the cached tensors. We can called this policy as `NO_EVICT` simply.
However, this policy may be not optimal sometimes. For example, generating
different constant tensor may have different cost (like consumed time, memory,
...), users may prioritize evicting those low cost constant tensors and caching
those high cost ones.

Based on above considerations, we have two options:
- option 1: provide different policy in API and extend the above capacity
  control API to allow users to set/get cache management policy. The API users
  could try different policies for different scenarios and select the best one.
  - Pros: Simpler for library, we only need to define those policy and no need
    to make choice. Users have more flexibility to chose the best policy.
  - Cons: Harder for users, they need to make choice and try different policies.

```cpp
typedef enum  {
    NO_EVICT,
    COST,
    ... // Others
} policy_t;

void set_constant_tensor_cache_capacity(dnnl::engine::kind eng_kind, size_t size, policy_t policy);
void get_constant_tensor_cache_capacity(dnnl::engine::kind eng_kind, size_t* size, policy_t * policy);
```

The environment variable value could be improved to `engine_kind:size:policy` as below:

```shell
export ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY=cpu:10240:NO_EVICT;gpu:2048:NO_EVICT
```

- option 2: implement different policy inside library, and library  will chose a
  proper policy by itself.
  - Pros: Simpler for users, they don't need to try and make choice.
  - Cons: Harder for library, we need to make choice, and less flexibility for users.

Currently, option 2 is preferred since it makes user easy. If users want to
explicitly control the policy, we can expose the policy in API then.

### Cache performance optimization
With the above unified constant tensor cache definition, the `get_or_add` API is
suggested to be called every time the backend want to use the constant tensor in
cache like the following:

```cpp
// Execution start
promise_t c_promise;
// Query the tensor from cache
value_t cached_value
        = cache.get_or_add(bkd_id, spec_key, size, c_promise.get_future());
bool is_from_cache = cached_value.valid();
if (is_from_cache) {
    // 1. Get out the cached memory
    cached_t c_buffer = cached_value.get();
    // 2. Use the c_buffer
} else {
    // 1. Allocate new memory
    cached_t c_buffer = std::make_shared<dnnl_constant_buffer_t>(
            size, p_engine_, g_alloc_);
    // 2. Compute the value of c_buffer
    // 3. Fulfill the promise
    c_promise.set_value(c_buffer);
}
// ...
// Execution end
```

Calling `get_or_add` in every execution may introduce overhead (such as hash
table lookup, threads synchronization, ...). We need to reduce that overhead if
it becomes model level performance bottleneck. Some options can be considered:

- option 1: Further optimize the current `get_or_add` api implementation, like
  using multi-level cache, more lightweight threads synchronization, ... To
  reduce the function calling overhead.
  - Pros: The interface is keep as simple as possible, and all backend could
    have unified usage pattern.
  - Cons: Performance improvement may be limited, since we need to keep the
    interface general.
- option 2: Extend the interfaces to allowing different programming model and
  give backend more flexibilities to optimize cache performance by themselves.
  - Pros: More flexible for each backend, more optimization opportunities and
    performance improvement may be more significant.
  - Cons: The interface becomes more complex, and the cache usage pattern in
    different backend may be diverse, the backend API needs more time to
    converge.

We usually think it's more critical to optimize the overhead when cache hit.

A recommendation is to add a `callback/notification` mechanism in unified cache,
so that backend user can further hold the queried constant tensor handle by
itself. When the tensor is still cached in unified cache, backend could directly
access the handle efficiently without querying the unified cache every time.
When the tensor is evicted from the unified cache, the callback should be called
from unified cache side to tell backend the tensor is evicted, and backend will
stop using the tensor. This propose could be treated as a specific case of above
option 2.

Currently, option 2 is preferred by backend users since it's more flexible.

Backend API and integration optimization should not break the frontend API,
unless there are new requests from framework users.

### Multiple devices
For multiple devices scenarios, users can still use the `void
set_constant_tensor_cache_capacity(dnnl::engine::kind eng_kind, size_t
capacity)` API to set capacity. But the specified capacity here is the
budget for each device instead of a total budget for all devices. And the
library will maintain dedicated cache for each device and manage them
separately.

The above proposal is based on a consideration: only limiting total budget for
all devices may lead to a situation that one device has been OOM since cached
too much constant tensor but the whole cache size still does not reach the
capacity.

If users want to set different capacity for different device, we need to let
users specify which device to be set through API.

oneDNN Graph API allows users to use multiple devices in two ways:
- create engine with a device `index`.
- create engine with a `sycl::device`.

So, the following frontend capacity control APIs can be added to specify constant
tensor cache capacity for each device:

```cpp
void set_constant_tensor_cache_capacity(dnnl::engine::kind eng_kind, int index, size_t size);
void get_constant_tensor_cache_capacity(dnnl::engine::kind eng_kind, int index, size_t* size);

// The sycl interop API
void sycl_interop::set_constant_tensor_cache_capacity(const sycl::device & dev, size_t size);
void sycl_interop::get_constant_tensor_cache_capacity(const sycl::device & dev, size_t* size);
```

The weight sharing rules in multiple devices scenarios

In some special cases, like dynamic shape, multiple different compiled
partitions may share same weight. Based on the different combination of device
and context used by each compiled partition, we have some general rules:
- same device, same context: the compiled partitions that use both same device
  and context are allowed to share same cached weight.
- same device, different context: sycl programming api doesn't allow access
  memory cross different context. So the compiled partitions that use different
  context are not allowed to share same cached weight, even they are using same
  device.
- different device, same context: based on above discussions, constant tensor
  cache is device specific, so the compiled partitions that use different device
  will lookup different cache and will not hit same cached weight. This makes
  sense, because accessing memory cross device is usually slower than accessing
  local memory, so it's not preferred.
- different device, different context: not sharing weight at all.

Based on above analysis, weight sharing cross compiled partitions will only
happens when those compiled partitions are using same device and same context.

Supporting multiple devices usage remains an open topics. Any suggestions or
ideas are welcome and we can discuss it in more details once real requests
emerged. Currently, to make the API simple, we prefer to still use the `void
set_constant_tensor_cache_capacity(dnnl::engine::kind eng_kind, size_t
capacity)` API to set same capacity for all devices of same engine kind. We can
extend the API in future once the requirement pops up.