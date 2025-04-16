# Vision Mamba Discretization Methods

This directory contains scripts for training and evaluating Vision Mamba models with different discretization methods. The original Vision Mamba uses Zero Order Hold (ZOH) discretization, but we have implemented additional methods which may improve model performance.

## Implemented Discretization Methods

1. **Zero Order Hold (ZOH)** - The original implementation

   - Discretizes the continuous system by holding the input constant over each time step
   - Simple but can introduce inaccuracies for rapidly changing signals

2. **First Order Hold (FOH)**

   - Uses linear interpolation between sample points
   - More accurate than ZOH, especially for smoothly varying signals
   - Formula: A_d = exp(A*delta), B_d = (A^-1)*(A_d - I)\*B

3. **Bilinear (Tustin) Transform**

   - Maps the s-plane to the z-plane using a trapezoidal approximation
   - Preserves stability properties of the original system
   - Formula: A*d = (I + A*delta/2)^(-1) * (I - A*delta/2), B_d = (I + A*delta/2)^(-1) * delta \_ B

4. **Polynomial Interpolation**

   - Uses higher-order polynomial approximation for the input between sample points
   - Provides better accuracy for complex signals
   - Formula: B_d = delta*B + delta^2*A*B/2 + delta^3*A^2\*B/6 (3rd order approximation)

5. **Higher-Order Hold**

   - Similar to FOH but with higher-order terms
   - Approximates using Taylor series expansion
   - Provides improved accuracy over ZOH and FOH

6. **Runge-Kutta 4th Order (RK4)**
   - Uses the classic 4th order Runge-Kutta method for numerical integration
   - Provides high accuracy and stability for complex dynamics
   - Formula:
     - k1 = delta \* f(t_n, y_n)
     - k2 = delta \* f(t_n + delta/2, y_n + k1/2)
     - k3 = delta \* f(t_n + delta/2, y_n + k2/2)
     - k4 = delta \* f(t_n + delta, y_n + k3)
     - y\_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4)/6
   - Advantages:
     - Higher accuracy (4th order) compared to other methods
     - Better stability properties
     - More accurate approximation of continuous-time dynamics
     - Proper handling of non-linearities through the RK4 coefficients
   - Trade-offs:
     - More computationally expensive (requires 4 function evaluations per step)
     - More complex implementation
     - May require smaller step sizes for stability

## Dataset Structure

The scripts are configured to use the following directories:

- Training data: `/Volumes/X10 Pro/datasets/imagenet-1k/train`
- Validation data: `/Volumes/X10 Pro/datasets/imagenet-1k/validation`

## Training Scripts

Each discretization method has its own training script:

- `pt-vim-zoh.sh` - Train with Zero Order Hold (default)
- `pt-vim-foh.sh` - Train with First Order Hold
- `pt-vim-bilinear.sh` - Train with Bilinear Transform
- `pt-vim-poly.sh` - Train with Polynomial Interpolation
- `pt-vim-highorder.sh` - Train with Higher-Order Hold
- `pt-vim-rk4.sh` - Train with Runge-Kutta 4th Order

### Running Full Training

To run a full training for a specific discretization method:

```bash
# Make the script executable
chmod +x scripts/pt-vim-foh.sh

# Run the training
./scripts/pt-vim-foh.sh
```

### Running All Methods Sequentially

To run all discretization methods one after another:

```bash
chmod +x scripts/run-all-discretization-methods.sh
./scripts/run-all-discretization-methods.sh
```

### Quick Testing

For quick testing with fewer resources:

```bash
chmod +x scripts/test-discretization-methods.sh
./scripts/test-discretization-methods.sh
```

## Evaluation and Comparison

After training, you can compare the performance of all methods using the comparison script:

```bash
# Make the script executable
chmod +x scripts/compare-discretization-methods.py

# Run the comparison
python scripts/compare-discretization-methods.py --output ./comparison_results
```

This will:

1. Load each trained model
2. Evaluate it on the validation set
3. Compare their performances
4. Generate plots and a CSV file with the results

## Expected Outcomes

Different discretization methods may perform better depending on the nature of the data and the specific task. Generally, we might expect:

- FOH and Bilinear methods may provide better stability for rapidly changing signals
- Higher-order methods might capture more complex patterns but could be computationally more expensive
- Polynomial interpolation might provide a good balance between accuracy and computational cost
- RK4 method may provide the highest accuracy but with increased computational overhead

## Modified Files

To implement these discretization methods, the following files were modified:

1. `mamba_ssm/ops/selective_scan_interface.py` - Added support for multiple discretization methods
2. `mamba_ssm/modules/mamba_simple.py` - Updated the Mamba class to support different discretization methods
3. `vim/models_mamba.py` - Added new model variants for each discretization method
