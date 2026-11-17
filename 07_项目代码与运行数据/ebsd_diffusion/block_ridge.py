"""CPU-only nested block-ridge diagnostics. No generator/training authorization.

Every learned transform (scaling, PCA, correlation pruning) is fitted inside
each inner training fold as well as the outer training fold. Descriptors and
composition have independent L2 penalties. An explicit 'off' image candidate
can be registered; it is not a guarantee that outer CV cannot get worse.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import spearmanr


MODES = ('composition', 'shared', 'block_pc1', 'block_all', 'image_only', 'hard')


def standardize_fit(x):
    mean, scale = x.mean(0), x.std(0)
    return mean, np.where(scale < 1e-12, 1., scale)


def rank_diagnostics(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Rank diagnostics require finite features')
    mean, scale = standardize_fit(x)
    s = np.linalg.svd((x-mean)/scale, compute_uv=False)
    if not len(s) or s[0] < 1e-12:
        return {'numerical_rank': 0, 'effective_rank': 0., 'condition_number': None,
                'singular_values': s.tolist()}
    p = s*s/(s*s).sum()
    return {'numerical_rank': int(np.linalg.matrix_rank((x-mean)/scale)),
            'effective_rank': float(np.exp(-np.sum(p[p > 0]*np.log(p[p > 0])))),
            'condition_number': float(s[0]/s[-1]) if s[-1] > 1e-12 else None,
            'singular_values': s.tolist()}


@dataclass
class Transform:
    c_mean: np.ndarray
    c_scale: np.ndarray
    d_mean: np.ndarray
    d_scale: np.ndarray
    retained: list[int]
    components: np.ndarray | None
    use_composition: bool

    @property
    def n_composition(self):
        return len(self.c_mean) if self.use_composition else 0

    def apply(self, c, d):
        c, d = np.atleast_2d(c), np.atleast_2d(d)
        cc = (c-self.c_mean)/self.c_scale if self.use_composition else c[:, :0]
        dd = (d[:, self.retained]-self.d_mean)/self.d_scale
        if self.components is not None:
            dd = dd @ self.components.T
        return np.column_stack([cc, dd])

    def metadata(self):
        return {'composition_mean': self.c_mean.tolist(), 'composition_scale': self.c_scale.tolist(),
                'image_mean': self.d_mean.tolist(), 'image_scale': self.d_scale.tolist(),
                'retained_image_columns': self.retained,
                'pca_components': None if self.components is None else self.components.tolist(),
                'use_composition': self.use_composition}


def fit_transform(c, d, mode, *, priority=None, correlation_threshold=.9, max_features=3):
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    c, d = np.asarray(c, float), np.asarray(d, float)
    if c.ndim != 2 or d.ndim != 2 or len(c) != len(d) or not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError('Expected paired finite composition/image matrices')
    cm, cs = standardize_fit(c)
    retained = list(range(d.shape[1])) if mode != 'composition' else []
    if mode == 'hard':
        order = list(range(d.shape[1])) if priority is None else list(priority)
        if sorted(order) != list(range(d.shape[1])):
            raise ValueError('Priority must list each feature exactly once')
        retained = []
        for index in order:
            if d[:, index].std() < 1e-12:
                continue
            if any(abs(float(np.corrcoef(d[:, index], d[:, previous])[0, 1])) > correlation_threshold
                   for previous in retained):
                continue
            retained.append(index)
            if len(retained) >= max_features:
                break
    dm, ds = standardize_fit(d[:, retained])
    components = None
    if mode == 'block_pc1':
        z = (d[:, retained]-dm)/ds
        if not z.shape[1]: raise ValueError('PC1 needs at least one image feature')
        _, _, vectors = np.linalg.svd(z, full_matrices=False)
        components = vectors[:1].copy()
        pivot = int(np.argmax(np.abs(components[0])))
        if components[0, pivot] < 0:
            components *= -1
    return Transform(cm, cs, dm, ds, retained, components, mode != 'image_only')


@dataclass
class BlockModel:
    transform: Transform
    coefficient: np.ndarray
    target_mean: float
    target_scale: float
    alpha: float
    image_ratio: float | None

    def predict(self, c, d):
        return self.target_mean+self.target_scale*(self.transform.apply(c, d) @ self.coefficient)

    def contributions(self, c, d):
        z = self.transform.apply(c, d)
        p = self.transform.n_composition
        composition = self.target_scale*(z[:, :p] @ self.coefficient[:p])
        image = self.target_scale*(z[:, p:] @ self.coefficient[p:])
        return {'intercept': self.target_mean, 'composition': composition.tolist(),
                'image': image.tolist(), 'prediction': self.predict(c, d).tolist()}

    def metadata(self):
        return {'transform': self.transform.metadata(), 'coefficient': self.coefficient.tolist(),
                'target_mean': self.target_mean, 'target_scale': self.target_scale,
                'alpha_composition': self.alpha,
                'image_penalty_ratio': 'off' if self.image_ratio is None else self.image_ratio}


def candidates(mode, alphas, ratios):
    if any(not np.isfinite(a) or a <= 0 for a in alphas) or not alphas:
        raise ValueError('Positive predeclared alphas required')
    pairs = []
    for alpha in alphas:
        if mode in ('composition', 'shared', 'image_only'):
            pairs.append((float(alpha), 1.))
        else:
            for ratio in ratios:
                if ratio == 'off': pairs.append((float(alpha), None))
                elif np.isfinite(ratio) and ratio > 0: pairs.append((float(alpha), float(ratio)))
                else: raise ValueError('Image ratios must be positive or off')
    if not pairs: raise ValueError('Empty candidate grid')
    return pairs


def coefficient_bank(z, y, n_composition, pairs):
    """Batched small linear solves; preprocessing is shared across candidates only within a fold."""
    y_mean, y_scale = float(y.mean()), float(y.std())
    y_scale = max(y_scale, 1e-12)
    yy = (y-y_mean)/y_scale
    dimension = z.shape[1]
    coefficients = np.zeros((len(pairs), dimension))
    active = [i for i, (_, ratio) in enumerate(pairs) if ratio is not None]
    if active and dimension:
        penalties = np.asarray([[a]*n_composition+[a*ratio]*(dimension-n_composition)
                                for a, ratio in (pairs[i] for i in active)])
        matrices = np.broadcast_to(z.T@z, (len(active), dimension, dimension)).copy()
        diagonal = np.arange(dimension)
        matrices[:, diagonal, diagonal] += penalties
        rhs = np.broadcast_to((z.T@yy)[None, :, None], (len(active), dimension, 1))
        coefficients[active] = np.linalg.solve(matrices, rhs)[..., 0]
    for i, (alpha, ratio) in enumerate(pairs):
        if ratio is None and n_composition:
            zz = z[:, :n_composition]
            coefficients[i, :n_composition] = np.linalg.solve(
                zz.T@zz+alpha*np.eye(n_composition), zz.T@yy)
    return coefficients, y_mean, y_scale


def tune(c, d, y, mode, alphas, ratios, transform_settings=None):
    if len(y) < 3: raise ValueError('Inner tuning requires at least three alloys')
    pairs = candidates(mode, alphas, ratios)
    errors = np.zeros(len(pairs))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        t = fit_transform(c[keep], d[keep], mode, **(transform_settings or {}))
        z = t.apply(c[keep], d[keep])
        coefficients, ym, ys = coefficient_bank(z, y[keep], t.n_composition, pairs)
        predictions = ym+ys*(coefficients @ t.apply(c[i:i+1], d[i:i+1])[0])
        errors += (predictions-y[i])**2
    winner = int(np.argmin(errors))
    return pairs[winner]


def fit_selected(c, d, y, mode, pair, transform_settings=None):
    t = fit_transform(c, d, mode, **(transform_settings or {}))
    coef, ym, ys = coefficient_bank(t.apply(c, d), y, t.n_composition, [pair])
    return BlockModel(t, coef[0], ym, ys, *pair)


def nested_predict(c, d, y, mode, alphas, ratios, *, transform_settings=None, return_models=False):
    c, d, y = np.asarray(c, float), np.asarray(d, float), np.asarray(y, float)
    if len(y) < 4 or not np.isfinite(y).all():
        raise ValueError('Outer validation requires >=4 finite alloy labels')
    predictions, models = [], []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        pair = tune(c[keep], d[keep], y[keep], mode, alphas, ratios, transform_settings)
        model = fit_selected(c[keep], d[keep], y[keep], mode, pair, transform_settings)
        predictions.append(float(model.predict(c[i:i+1], d[i:i+1])[0]))
        models.append(model)
    return (np.asarray(predictions), models) if return_models else np.asarray(predictions)


def scores(y, prediction):
    residual = np.asarray(prediction)-y
    correlation = None
    if np.std(y) > 1e-12 and np.std(prediction) > 1e-12:
        correlation = float(spearmanr(y, prediction).statistic)
    return {'rmse': float(np.sqrt(np.mean(residual**2))), 'mae': float(np.mean(np.abs(residual))),
            'spearman': correlation}


def validate_representation(c, d, y, ids, mode, *, alphas, ratios, transform_settings=None,
                            permutations=1000, seed=20261102, non_degradation=True,
                            permutation_p_threshold=.05):
    """All original folds and delete-one repeats are retained; no posthoc ID removal."""
    if len(y) < 5: raise ValueError('Delete-one nested validation requires >=5 alloys')
    c, d, y, ids = np.asarray(c, float), np.asarray(d, float), np.asarray(y, float), np.asarray(ids)
    base = nested_predict(c, d, y, 'composition', alphas, ratios)
    prediction, models = nested_predict(c, d, y, mode, alphas, ratios,
                                       transform_settings=transform_settings, return_models=True)
    sb, sj = scores(y, base), scores(y, prediction)
    gain = 1-sj['rmse']/sb['rmse'] if sb['rmse'] > 1e-12 else 0.
    deletion = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        b = nested_predict(c[keep], d[keep], y[keep], 'composition', alphas, ratios)
        j = nested_predict(c[keep], d[keep], y[keep], mode, alphas, ratios,
                           transform_settings=transform_settings)
        deletion.append(float(np.mean((b-y[keep])**2)-np.mean((j-y[keep])**2)))
    improved = int(np.sum(np.abs(prediction-y) < np.abs(base-y)))
    primary = bool(gain >= .15 and improved >= math.ceil(12*len(y)/21) and min(deletion) > 0)
    additional = bool(sj['mae'] <= sb['mae'] and sj['spearman'] is not None and sb['spearman'] is not None
                      and sj['spearman'] >= sb['spearman'])
    rng = np.random.default_rng(seed)
    observed = sb['rmse']**2-sj['rmse']**2
    null_gains = []
    for _ in range(permutations):
        shuffled = d[rng.permutation(len(d))]
        jp = nested_predict(c, shuffled, y, mode, alphas, ratios, transform_settings=transform_settings)
        null_gains.append(float(sb['rmse']**2-np.mean((jp-y)**2)))
    tail_probability = (1+sum(v >= observed for v in null_gains))/(permutations+1)
    prediction_passed = bool(primary and (additional or not non_degradation))
    permutation_passed = bool(permutations >= 1000 and tail_probability <= permutation_p_threshold)
    return {'mode': mode, 'alloy_ids': ids.tolist(), 'composition_scores': sb, 'model_scores': sj,
            'relative_rmse_gain': gain, 'improved_alloys': improved, 'required_improved': math.ceil(12*len(y)/21),
            'composition_oof': base.tolist(), 'model_oof': prediction.tolist(),
            'delete_one_mse_gains': deletion, 'original_prediction_gate_passed': primary,
            'additional_non_degradation_passed': additional,
            'v2_prediction_gate_passed': prediction_passed,
            'permutation_count': permutations,
            'permutation_tail_probability': tail_probability,
            'permutation_p_threshold': permutation_p_threshold,
            'permutation_gate_passed': permutation_passed,
            'cpu_proxy_gate_passed': bool(prediction_passed and permutation_passed),
            'permutation_interpretation': 'pairing sensitivity; not conditional independence',
            'outer_fold_models': [m.metadata() for m in models]}


def composition_to_pc1(c, d, alphas):
    """Fold-local PC targets: pooled score uses each training-fold mean as reference."""
    c, d = np.asarray(c, float), np.asarray(d, float)
    observed, predicted, zero = [], [], []
    for i in range(len(c)):
        keep = np.arange(len(c)) != i
        cc, dd = c[keep], d[keep]
        grid_error = np.zeros(len(alphas))
        for j in range(len(cc)):
            inner = np.arange(len(cc)) != j
            t = fit_transform(cc[inner], dd[inner], 'block_pc1')
            pc = t.apply(cc[inner], dd[inner])[:, -1]
            target = t.apply(cc[j:j+1], dd[j:j+1])[0, -1]
            ct = fit_transform(cc[inner], dd[inner], 'composition')
            coef, ym, ys = coefficient_bank(ct.apply(cc[inner], dd[inner]), pc, cc.shape[1],
                                            [(a, 1.) for a in alphas])
            pp = ym+ys*(coef @ ct.apply(cc[j:j+1], dd[j:j+1])[0])
            grid_error += (pp-target)**2
        t = fit_transform(cc, dd, 'block_pc1')
        pc = t.apply(cc, dd)[:, -1]
        target = t.apply(c[i:i+1], d[i:i+1])[0, -1]
        ct = fit_transform(cc, dd, 'composition')
        coef, ym, ys = coefficient_bank(ct.apply(cc, dd), pc, cc.shape[1], [(alphas[int(np.argmin(grid_error))], 1.)])
        observed.append(float(target)); predicted.append(float(ym+ys*(coef[0] @ ct.apply(c[i:i+1], d[i:i+1])[0])))
        zero.append(float(pc.mean()))
    denom = float(np.sum((np.asarray(observed)-zero)**2))
    return {'observed_fold_local_pc1': observed, 'predicted': predicted,
            'cv_r2_vs_fold_training_mean': 1-float(np.sum((np.asarray(predicted)-observed)**2))/denom if denom > 1e-12 else None,
            'limits': 'PC basis is fitted independently within each fold; not a mediation/causal test'}


def influence_diagnostics(model, c, d, y, query_c=None, query_d=None):
    """Conditional ridge leverage/Cook-like diagnostic, not classical OLS inference."""
    z = model.transform.apply(c, d)
    pc = model.transform.n_composition
    if model.image_ratio is None:
        z = z[:, :pc]
        penalty = np.full(pc, model.alpha)
    else:
        penalty = np.asarray([model.alpha]*pc+[model.alpha*model.image_ratio]*(z.shape[1]-pc))
    inverse = np.linalg.inv(z.T@z+np.diag(penalty))
    h = np.ones((len(z), len(z)))/len(z)+z@inverse@z.T
    residual = np.asarray(y)-model.predict(c, d)
    edf = float(np.trace(h))
    variance = float(residual@residual/(len(z)-edf)) if len(z)-edf > 1e-10 else None
    cooks = []
    for i in range(len(z)):
        denom = 1-h[i, i]
        delta = h[:, i]*residual[i]/denom if abs(denom) > 1e-12 else None
        cooks.append(float(delta@delta/(edf*variance)) if delta is not None and variance and edf > 0 else None)
    query = None
    if query_c is not None:
        q = model.transform.apply(query_c, query_d)
        if model.image_ratio is None: q = q[:, :pc]
        query = (1/len(z)+np.einsum('bi,ij,bj->b', q, inverse, q)).tolist()
    return {'training_hat_diagonal': np.diag(h).tolist(), 'effective_df': edf,
            'conditional_residual_variance_estimate': variance, 'ridge_cook_like': cooks,
            'query_design_leverage': query,
            'limits': 'Conditional on fitted scaling/PCA/penalties; not an OLS significance test or a deletion-based admission rule'}
