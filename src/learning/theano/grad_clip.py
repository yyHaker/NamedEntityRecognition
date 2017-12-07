import theano.tensor as T
import theano

# test gradient
x = theano.tensor.scalar()

z = theano.tensor.grad(theano.gradient.grad_clip(x, -1, 1) ** 2, x)
z2 = theano.tensor.grad(x ** 2, x)

f = theano.function([x], outputs=[z, z2])

print(f(0.4))  # output (1.0, 4.0)
