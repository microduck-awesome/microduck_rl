"""Review all24 consecutive common-budget snapshots with the frozen evaluator."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from servo_bam.testbench.learning_evidence import digest, require
from servo_bam.testbench.learning_review import review_window


METRICS = ['mean_episode_MAE_deg', 'mean_episode_P99_deg', 'mean_zero_hold_std_deg']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve previous analyses; use a fresh output directory')
    protocol = json.loads(args.protocol.read_text())
    protocol_sha = digest(args.protocol)
    rules = protocol['convergence']
    require(len(args.reports) == rules['window_evaluations'],
            'Supply the complete declared consecutive review window')
    snapshots = [json.loads(path.read_text()) for path in args.reports]
    budgets = [snapshot['common_completed_iterations'] for snapshot in snapshots]
    require(all(b - a == rules['evaluation_interval'] for a, b in zip(budgets, budgets[1:])),
            'Common budgets must be consecutive and supplied in chronological order')
    require(budgets[-1] >= rules['minimum_iterations'], 'The initial minimum budget is incomplete')
    expected = {(p, seed) for p in protocol['policy_groups'] for seed in protocol['seeds']}
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen source/input changed: ' + path)
    validator = Path(__file__).with_name('sc0090_validate_checkpoint.py').resolve()
    executor_sha = protocol['source_sha256'][str(validator)]
    for snapshot in snapshots:
        require(snapshot['passed'] and snapshot['all24_current_executor_validations']
                and snapshot['evaluation_protocol_sha256'] == protocol_sha
                and snapshot['validation_executor_sha256'] == executor_sha,
                'A snapshot did not use the current frozen evaluation contract')
        require(len(snapshot['policies']) == 24
                and {(row['policy'], row['seed']) for row in snapshot['policies']} == expected,
                'A snapshot does not contain exactly the full24 identities')
        for path, sha in snapshot['evidence_sha256'].items():
            require(digest(path) == sha, 'Bound snapshot evidence changed: ' + path)
    rows = []
    for policy in protocol['policy_groups']:
        for seed in protocol['seeds']:
            checkpoints = [next(row for row in snapshot['policies']
                                if row['policy'] == policy and row['seed'] == seed)
                           for snapshot in snapshots]
            # Checkpoints retain their original training identities. Analysis groups
            # them by the common evaluation contract, not by a relabeled training one.
            analysis = [{**row, 'protocol': str(args.protocol.resolve()),
                         'protocol_sha256': protocol_sha, 'reward_mean': None}
                        for row in checkpoints]
            whole = review_window(analysis, len(analysis), 2048, 91141)
            latest = review_window(analysis, 2, 2048, 91141)
            require(whole['measured'] and latest['measured'], 'An actor window could not be measured')
            rows.append({'policy': policy, 'seed': seed,
                'checkpoint_sha256': [row['checkpoint_sha256'] for row in checkpoints],
                'training_protocol_sha256': [row['training_protocol_sha256'] for row in checkpoints],
                'mae_deg': [row['mae_deg'] for row in checkpoints],
                'pooled_p99_deg': [row['p99_deg'] for row in checkpoints],
                'zero_mae_deg': [row['zero_mae_deg'] for row in checkpoints],
                'zero_hold_std_deg': [row['zero_hold_std_deg'] for row in checkpoints],
                'whole_window': whole, 'latest_interval': latest})
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen source/input changed during analysis: ' + path)
    counts = {scope: {metric: dict(Counter(next(item['observed_change']
        for item in row[scope]['measured_learning_changes'] if item['metric'] == metric)
        for row in rows)) for metric in METRICS} for scope in ['whole_window', 'latest_interval']}
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'source.py')
    import servo_bam.testbench.learning_review as dependency
    for name in ['learning_review.py', 'learning_evidence.py']:
        shutil.copyfile(Path(dependency.__file__).with_name(name), args.output / name)
    for budget, path in zip(budgets, args.reports, strict=True):
        shutil.copyfile(path, args.output / f'common{budget:04}-snapshot.json')
    result = {'passed': True, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'common_budgets': budgets, 'policies': rows, 'classification_counts': counts,
        'source_reports_sha256': {str(path.resolve()): digest(path) for path in args.reports},
        'source_sha256': digest(__file__), 'review_dependency_sha256': digest(dependency.__file__),
        'evaluation_protocol_sha256': protocol_sha, 'validation_executor_sha256': executor_sha,
        'frozen_contract_files_unchanged': len(frozen), 'paired_worlds_per_actor': 320,
        'bootstrap_draws': 2048, 'formal_convergence_claimed': False,
        'automatic_stopping_decision': False, 'fixed_degree_threshold_selected': False,
        'hardware_executed': False, 'sim2real_benefit_verified': False,
        'scope': 'Four consecutive same-executor common-budget validations for every policy/seed. '
                 'Whole-trajectory paired sampling intervals across13 metrics within each actor. '
                 'Original training identities and R1 prefix provenance remain explicit. '
                 'No across-policy significance, fitted-model uncertainty, physical uncertainty, '
                 'equivalence, optimality or automatic stopping conclusion.'}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    labels = {'improving': '改善', 'regressing': '退步', 'unresolved_at_this_precision': '变化未明确'}
    text = ['# 全24当前验证版本连续学习窗口', '',
        '共同保存轮数：' + '、'.join(map(str, budgets)) + '。所有指标来自同一冻结验证器；每份320条完整8秒轨迹。', '',
        '| 策略 | 种子 | MAE序列/度 | 全窗口MAE | 最新MAE | 全窗口每轨迹P99 | 全窗口静止抖动 |',
        '| --- | ---: | --- | --- | --- | --- | --- |']
    for row in rows:
        classifications = {scope: {item['metric']: labels[item['observed_change']]
                           for item in row[scope]['measured_learning_changes']}
                           for scope in ['whole_window', 'latest_interval']}
        text.append('| ' + ' | '.join([row['policy'], str(row['seed']),
            ' → '.join(f'{value:.4f}' for value in row['mae_deg']),
            classifications['whole_window'][METRICS[0]], classifications['latest_interval'][METRICS[0]],
            classifications['whole_window'][METRICS[1]], classifications['whole_window'][METRICS[2]]]) + ' |')
    text.extend(['', '全窗口趋势与最新区间变化分别报告，避免用早期改善掩盖近期停滞或退步。变化未明确不代表等价或收敛。', '',
        '13项指标内部同时95%区间仅描述每份策略固定仿真条件的样本变化；每轨迹P99均值与总体P99另行保留。'
        '停止收益还需结合当前任务、当前权重的重复测量和连续质量检查；这里未指定固定角度阈值，也未停止训练。', ''])
    (args.output / 'REVIEW.md').write_text('\n'.join(text))
    print(json.dumps({'passed': True, 'common_budgets': budgets, 'measured_policy_windows': len(rows),
                      'classification_counts': counts, 'formal_convergence_claimed': False}))


if __name__ == '__main__':
    main()
