# Proposal to calculate mean & variance in batchnorm in single pass

## Requester
GPU Enabling team

## Motivation
Using new formula for calculating variance will result in up to 33% performance
improvement in batchnorm fwd training operation. 

Currently bnorm performance is far below projections. Projections are based on
assumption that bnorm reads input tensor once and writes result once. Current
implementation requires bnorm to read input tensor 3 times. Proposed change
will decrease it to 2 reads. There is no known generic algorithm to decrease it
to 1 read, although such efficiency is possible for special cases where input
tensor is small enough.

## Proposal
Use the following formula to calculate variance:
```
Var(X) = E(X^2) - (E(X))^2
```
where X = input tensor, E = average

This formula allows hardware to calculate both mean and variance in single pass
over input tensor.
For reference, the canonical formula is:
```
Var(X) = E((X - Mean(X))^2)
```
The benefit of new formula is 25-33% performance improvement. Batchnorm kernel
is memory-bound, with new formula it will need to read input tensor 2 times 
(1: mean&var, 2: normalize) instead of 3 times (1: mean, 2: var, 3: normalize)

This formula will be used whenever batchnorm kernel calculates mean and 
variance, which means forward propagation in training. However, with 
non-standard combination of flags, oneDNN's batchnorm implementaton may be
configured to calculate those two values in inference as well.

## Design
Algorithm for calculating variance will be implemented as:
```
Sum = 0
SumOfSquares = 0
for each x in X:
	Sum += x
	SumOfSquares += (x*x)
Variance = SumOfSquares/N - (Sum/N)^2
if Variance < 0 then Variance = 0
```
where N = number of items in given channel in tensor.

In the formula there's a subtraction of two large values which is vulnerable
to catastrophic cancellation. Rounding errors in the two big numbers may be
large in relation to result of the formula. In order to minimize rounding
errors:
- Sum and SumOfSquares will be stored in fp32 precision, regardless of input
	data type
- Kahan summation algorithm will be used

The last line, "if var < 0 then var = 0" is there to make sure rounding errors
won't cause variance to go negative. Further in batchnorm calculations square
root is taken from variance. Negative value would result in NaN.

New formula should be considered experimental. By default the old one will be
used. There needs to be a way for user to choose formula. It is not decided yet
how exactly user will toggle it.

### Enable through environmental variable
OneDNN will read env var `DNNL_EXPERIMENTAL_BNORM_ONE_PASS`. If not set or
equal to zero, old formula will be used.

Optionally, as a safety measure against accidentally leaving the above flag 
in wrong state, add a `DNNL_EXPERIMENTAL` build knob (disabled by default). The
above env var would only matter when oneDNN was built with this knob enabled.

Pros:
- API/ABI compatibility
- Simplest to implement and use
- Clearly defined default state
- If needed, may be kept confidential and not disclosed to external customes 
until tested internally

Cons:
- No fine-tuning per instance of bnorm primitive
- Harder to reproduce bugs related to bnorm: value of this flag will need to be
 reported in bug reports to allow reproduction.
 - Inconvenient long-term. State of oneDNN will have a dependency which is not
 obvious to users.

### Enable through additional flag in bnorm primitive descriptor
Bnorm descriptor has flags field `dnnl_normalization_flags_t`. Add new flag 
`dnnl_one_pass = 0x20u`. When set to non-zero, new formula will be used.

Pros:
- Fine-tuning per instance of bnorm primitive
- Suitable to become permanent solution, should both formulas be kept in oneDNN

Cons:
- API change needed.
- In case a customer was passing garbage into bit5 of flags, the default
state is undefined and behavior may unexpectedly change.
- If we decide to remove it later and keep only one formula, it will require
another API change; it will leave behind a deprecated flag.
- Inconsistency: not all implementations of bnorm primitive support that flag.
Specifically, CPU implementation doesn't have it.
- Inconvenient for validation. In order to test this feature with a topology,
code change in frameworks in needed as frameworks will need to set this flag 
for each bnorm instance.

### Enable through new API call
Introduce new API function `void set_bnorm_one_pass_mode(bool onepass)`. This
function will set global variable which will subsequently control behavior of
all instances of bnorm primitive.

Pros:
- API stays backwards-compatible
- Clearly defined default state

Cons:
- API change needed.
- No fine-tuning per instance of bnorm primitive
- If we decide to remove it later and keep only  one formula, it will require
 another API change; it will leave behind a deprecated function.

## Validation
New formula was tested with RN50 and it trained to SotA. Further tests with
broader set of topologies are needed. Until such tests are performed, new
formula should be optionally available but disabled by default. The point
of this request is to make new formula available for further validation, while
external customers keep using the old and proven formula.
