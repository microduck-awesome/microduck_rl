"""Measure all24 paired learning changes between current-executor common budgets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from servo_bam.testbench.learning_evidence import digest, require
from servo_bam.testbench.learning_review import review_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--earlier', type=Path, required=True)
    parser.add_argument('--later', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a fresh paired learning report directory')
    snapshots = [json.loads(path.read_text()) for path in [args.earlier, args.later]]
    protocol = json.loads(args.protocol.read_text())
    protocol_sha = digest(args.protocol)
    require(all(s['passed'] and s['all24_current_executor_validations']
                and s['evaluation_protocol_sha256'] == protocol_sha for s in snapshots),
            'Both common-budget reports must contain all24 current-executor validations')
    require(len({s['validation_executor_sha256'] for s in snapshots}) == 1,
            'The evaluation executor changed across budgets')
    budgets = [s['common_completed_iterations'] for s in snapshots]
    require(budgets[1] - budgets[0] == protocol['convergence']['evaluation_interval'],
            'Compare two consecutive declared common budgets')
    for snapshot in snapshots:
        for path, sha in snapshot['evidence_sha256'].items():
            require(digest(path) == sha, 'A bound common-budget actor/trace/report changed: ' + path)
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen current source/input changed: ' + path)
    expected = {(p, seed) for p in protocol['policy_groups'] for seed in protocol['seeds']}
    for snapshot in snapshots:
        require(len(snapshot['policies']) == 24
                and {(p['policy'], p['seed']) for p in snapshot['policies']} == expected,
                'Common-budget identities differ from the full24 protocol')
    rows = []
    for policy in protocol['policy_groups']:
        for seed in protocol['seeds']:
            pair = [next(p for p in s['policies'] if p['policy'] == policy and p['seed'] == seed)
                    for s in snapshots]
            analysis_rows = [{**row, 'protocol': str(args.protocol.resolve()),
                              'protocol_sha256': protocol_sha, 'reward_mean': None} for row in pair]
            change = review_window(analysis_rows, 2, 2048, 91141)
            require(change['measured'] and change['fixed_target_family_condition_profiles'],
                    'Matched whole-trajectory change could not be measured')
            rows.append({'policy': policy, 'seed': seed, 'from_iterations': budgets[0],
                'to_iterations': budgets[1], 'mae_deg': [p['mae_deg'] for p in pair],
                'pooled_p99_deg': [p['p99_deg'] for p in pair],
                'zero_mae_deg': [p['zero_mae_deg'] for p in pair],
                'zero_hold_std_deg': [p['zero_hold_std_deg'] for p in pair],
                'checkpoint_sha256': [p['checkpoint_sha256'] for p in pair],
                'training_protocol_sha256': [p['training_protocol_sha256'] for p in pair],
                'evaluation_protocol_sha256': protocol_sha, 'paired_change': change})
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'source.py')
    import servo_bam.testbench.learning_review as dependency
    for name in ['learning_review.py', 'learning_evidence.py']:
        shutil.copyfile(Path(dependency.__file__).with_name(name), args.output / name)
    for name, path in [('earlier-snapshot.json', args.earlier), ('later-snapshot.json', args.later)]:
        shutil.copyfile(path, args.output / name)
    result = {'passed': True, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'common_budgets': budgets, 'policies': rows,
        'source_reports_sha256': {str(path.resolve()): digest(path) for path in [args.earlier, args.later]},
        'source_sha256': digest(__file__), 'review_dependency_sha256': digest(dependency.__file__),
        'evaluation_protocol_sha256': protocol_sha,
        'validation_executor_sha256': snapshots[0]['validation_executor_sha256'],
        'frozen_contract_files_unchanged': len(frozen), 'paired_worlds_per_actor': 320,
        'bootstrap_draws': 2048, 'formal_convergence_claimed': False,
        'automatic_stopping_decision': False, 'fixed_degree_threshold_selected': False,
        'hardware_executed': False, 'sim2real_benefit_verified': False,
        'scope': 'Current-executor paired whole-trajectory learning changes within each policy/seed. '
                 'Each policy has simultaneous95% intervals across13 metrics; no across-policy significance '
                 'or hardware uncertainty claim. One100-iteration change is insufficient to establish '
                 'a sustained plateau or convergence. Original checkpoint training identities are preserved.'}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    labels = {'improving': '改善', 'regressing': '退步', 'unresolved_at_this_precision': '变化未明确'}
    text = [f'# 全24共同{budgets[0]}→{budgets[1]}轮配对变化', '',
        '同一修复后的验证器、每份320个完整8秒轨迹。逐策略按环境种子和轨迹族分层的整轨迹配对重采样2048次，13项指标内部同时95%区间。原模型训练来源保持不变。', '',
        '| 策略 | 种子 | MAE/度 | MAE变化 | 每轨迹P99变化 | 静止抖动变化 |',
        '| --- | ---: | --- | --- | --- | --- |']
    for row in rows:
        metrics = {r['metric']: r for r in row['paired_change']['measured_learning_changes']}
        classifications = [labels[metrics[key]['observed_change']] for key in
                           ['mean_episode_MAE_deg', 'mean_episode_P99_deg', 'mean_zero_hold_std_deg']]
        text.append('| ' + ' | '.join([row['policy'], str(row['seed']),
            f"{row['mae_deg'][0]:.4f} → {row['mae_deg'][1]:.4f}", *classifications]) + ' |')
    text.extend(['',
        '变化未明确不能当作等价或收敛。单个100轮变化不足以证明持续平台；重复性与连续窗口仍需检查。每轨迹P99均值与总体采样点P99是不同指标，两个原始总体值保存在report.json中。', '',
        '这些区间仅描述各策略固定仿真条件中的样本变化，不包含M6/S/O拟合误差或实物不确定性，也没有对全部24策略作同时显著性保证。未指定停止阈值，未自动停止训练。', ''])
    (args.output / 'REVIEW.md').write_text('\n'.join(text))
    print(json.dumps({'passed': True, 'common_budgets': budgets, 'paired_policy_seeds': len(rows),
                      'formal_convergence_claimed': False, 'automatic_stopping_decision': False}))


if __name__ == '__main__':
    main()
