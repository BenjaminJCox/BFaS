Working through "Bayesian Filtering and Smoothing" and implementing some of it in Julia (from the original MATLAB code)

Algorithms will hopefully be implemented in such a way as to take advantage of the features in Julia (auto-differentiation, functions as arguments)

Eventually filters will be wrapped up into linear and non-linear classes with a common interface within each class. Something analogous will be done for smoothers, although due to the nature of smoothers it will necessitate a different approach. If I feel really cool I might create a 'real time' interface for filtering, that is filtering data as it is coming in (I know this is 'easy' but it is somewhat beyond the scope of this).
