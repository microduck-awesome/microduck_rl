"""Bind all24 actors at one saved budget to the same current evaluation contract."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from servo_bam.testbench.learning_evidence import digest, require
from sc0090_revalidate_checkpoints import recompute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--revalidation-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Use a new common-budget report directory')
    history = json.loads(args.history_report.read_text())
    protocol = json.loads(args.protocol.read_text())
    current_sha = digest(args.protocol)
    require(history['passed'] and len(history['policies']) == 24, 'Incomplete verified full24 ancestry')
    for path, sha in history['evidence_sha256'].items():
        require(digest(path) == sha, 'Historical checkpoint evidence changed: ' + path)
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen active source/input changed: ' + path)
    validator = Path(__file__).with_name('sc0090_validate_checkpoint.py').resolve()
    executor_sha = digest(validator)
    require(executor_sha == protocol['source_sha256'][str(validator)], 'Different current executor')
    evidence = {str(args.history_report.resolve()): digest(args.history_report),
                str(args.protocol.resolve()): current_sha}
    refreshed = {}
    for path in args.revalidation_reports:
        result = json.loads(path.read_text())
        require(result['passed'] and result['evaluation_protocol_sha256'] == current_sha
                and result['validator_sha256'] == executor_sha, 'Different or failed revalidation contract')
        evidence[str(path.resolve())] = digest(path)
        for row in result['checkpoints']:
            key = row['checkpoint_sha256']
            require(key not in refreshed, 'Duplicate revalidation actor')
            refreshed[key] = row
    rows, maximum_difference = [], 0.0
    for policy in history['policies']:
        selected = [row for row in policy['history'] if row['completed_iterations'] == args.iterations]
        require(len(selected) == 1, 'A declared common-budget checkpoint is missing')
        original = selected[0]
        key = original['checkpoint_sha256']
        if original['passed'] is not True:
            replacement = refreshed.get(key, {})
            require(original['passed'] is None and original.get('validation_state') == 'not_completed'
                    and original['validation_report'] is None
                    and replacement.get('first_completed_validation_of_saved_actor') is True
                    and replacement.get('original_validation_passed') is None
                    and replacement.get('original_validation_report') is None
                    and replacement.get('interrupted_export_parity', {}).get('nodes') == 512,
                    'Missing historical validation requires an explicit completed saved-actor evaluation')
        checkpoint = Path(original['checkpoint'])
        require(digest(checkpoint) == key, 'Common-budget actor changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        info = state['infos']
        require(state['iter'] == args.iterations - 1
                and info['policy'] == policy['policy'] and info['seed'] == policy['seed']
                and info['training_protocol_sha256'] == original['protocol_sha256']
                and info['env_state']['common_step_counter'] == args.iterations * protocol['steps_per_iteration'],
                'Common-budget saved training identity changed')
        report_path = Path(original['validation_report']) if original['validation_report'] else None
        original_report = json.loads(report_path.read_text()) if report_path else None
        if (original_report and original_report['protocol_sha256'] == current_sha
                and original_report['source_sha256'] == executor_sha):
            role = 'original_current_executor'
            expected_onnx_sha = original_report['onnx_sha256']
        else:
            require(key in refreshed, 'Common-budget actor needs current-executor revalidation')
            replacement = refreshed[key]
            require(replacement['policy'] == policy['policy'] and replacement['seed'] == policy['seed']
                    and replacement['completed_iterations'] == args.iterations
                    and replacement['training_protocol_sha256'] == original['protocol_sha256'],
                    'Revalidation identity differs from the common-budget actor')
            report_path = Path(replacement['validation_report'])
            require(digest(report_path) == replacement['validation_report_sha256'], 'Refreshed report changed')
            expected_onnx_sha = original_report['onnx_sha256'] if original_report else replacement['onnx_sha256']
            require(digest(replacement['onnx']) == expected_onnx_sha, 'Bound saved export changed')
            role = ('saved_actor_current_executor_revalidation' if original_report else
                    'first_current_executor_validation_of_saved_actor')
        report = json.loads(report_path.read_text())
        require(report['checkpoint_sha256'] == key and report['policy'] == policy['policy']
                and report['protocol_sha256'] == current_sha and report['source_sha256'] == executor_sha
                and report['onnx_sha256'] == expected_onnx_sha,
                'Common-budget report changed actor/export/evaluation identity')
        difference = recompute(report_path, protocol)
        maximum_difference = max(maximum_difference, difference)
        number = int(policy['policy'][1:])
        row = {'policy': policy['policy'], 'seed': policy['seed'], 'completed_iterations': args.iterations,
            'S': f'S{number % 2}', 'O': f'O{number // 2 % 2}', 'R': f'R{number // 4}',
            **{key: report[key] for key in ['passed', 'fault_count', 'mae_deg', 'p99_deg', 'zero_hold_std_deg']},
            'zero_mae_deg': report['families']['zero']['mae_deg'],
            'checkpoint': str(checkpoint), 'checkpoint_sha256': key,
            'training_protocol_sha256': original['protocol_sha256'],
            'original_validation_passed': original['passed'],
            'evaluation_protocol_sha256': current_sha, 'validation_executor_sha256': executor_sha,
            'validation_role': role, 'validation_report': str(report_path),
            'validation_report_sha256': digest(report_path), 'raw_sha256': report['raw_sha256'],
            'onnx_sha256': report['onnx_sha256']}
        evidence.update({str(checkpoint): key, str(report_path): digest(report_path),
                         str(report_path.parent / 'traces.npz'): report['raw_sha256']})
        rows.append(row)
    expected = {(p, seed) for p in protocol['policy_groups'] for seed in protocol['seeds']}
    require(len(rows) == 24 and {(r['policy'], r['seed']) for r in rows} == expected,
            'Missing or duplicate common-budget policy/seed')
    rows.sort(key=lambda row: (row['policy'], protocol['seeds'].index(row['seed'])))
    summaries = []
    for policy in protocol['policy_groups']:
        selected = [r for r in rows if r['policy'] == policy]
        values = np.array([r['mae_deg'] for r in selected])
        summaries.append({'policy': policy, 'seeds': [r['seed'] for r in selected],
            'mae_three_seed_mean_deg': float(values.mean()),
            'mae_three_seed_sample_std_deg': float(values.std(ddof=1)),
            'mae_three_seed_range_deg': [float(values.min()), float(values.max())],
            'interpretation': 'Descriptive variation over the three trained seeds in this route; not a hardware confidence interval.'})
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen active source/input changed during analysis: ' + path)
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / 'source.py')
    shutil.copyfile(Path(__file__).with_name('sc0090_revalidate_checkpoints.py'), output / 'raw-score-helper.py')
    shutil.copyfile(args.protocol, output / 'evaluation-protocol.json')
    result = {'passed': True, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'common_completed_iterations': args.iterations, 'policies': rows, 'three_seed_summaries': summaries,
        'evaluation_protocol_sha256': current_sha, 'validation_executor_sha256': executor_sha,
        'all24_current_executor_validations': True, 'raw_metrics_independently_recomputed_per_actor': 13,
        'maximum_raw_score_difference_deg': maximum_difference, 'evidence_sha256': evidence,
        'frozen_contract_files_unchanged': len(frozen), 'source_sha256': digest(__file__),
        'raw_score_helper_sha256': digest(Path(__file__).with_name('sc0090_revalidate_checkpoints.py')),
        'final_checkpoint_selection_performed': False, 'formal_convergence_claimed': False,
        'hardware_executed': False, 'sim2real_benefit_verified': False,
        'scope': 'Full24 actors at a single saved training budget, evaluated by the same current executor and frozen validation rules. Training origins remain explicit. Each route retains its own training-side measurement/randomization conditions; this is monitoring, not cross-model or physical validation, and not a stopping decision.'}
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text = [f'# 全24共同{args.iterations}轮仿真检查', '',
        '全部24份使用同一修复后的验证器，每份两个环境种子、320个8秒轨迹，故障数均为零；13项原始分数独立复算。MAE单位为度。原训练权重和协议身份保持不变。', '',
        '| 策略 | S/O/R | seed42 | seed7 | seed1 | 三种子均值 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for summary in summaries:
        selected = [r for r in rows if r['policy'] == summary['policy']]
        first = selected[0]
        values = [f"{r['mae_deg']:.4f}" for r in selected]
        text.append('| ' + ' | '.join([summary['policy'], '/'.join(first[k] for k in ['S', 'O', 'R']),
                                      *values, f"{summary['mae_three_seed_mean_deg']:.4f}"]) + ' |')
    text.extend(['',
        'S0/S1为简单/识别测量模型，O0/O1为LS基线/完整观测器，R0/R1为名义条件/域随机化。三种子均值与标准差是描述性统计，不代表实际台架不确定性。', '',
        '此次仅统一保存预算与验证版本，各组仍在自身训练路线的模拟测量与随机化条件中检查；不能以此表直接宣称某组实机胜出。全部R1共同保留的v3前200轮来源仍明确记录，不宣称旧阶段等价。', '',
        f'{args.iterations}轮不是停止阈值或收敛证明；仍须同版连续学习窗口和当前任务重复性测量判断继续训练的收益。P99、静止偏差和静止抖动一并保存在metrics.csv，不能只依据MAE决定成功。', ''])
    (output / 'COMPARISON.md').write_text('\n'.join(text))
    print(json.dumps({'passed': True, 'common_iterations': args.iterations,
        'actors': len(rows), 'maximum_raw_score_difference_deg': maximum_difference,
        'three_seed_means': {s['policy']: s['mae_three_seed_mean_deg'] for s in summaries},
        'formal_convergence_claimed': False}))


if __name__ == '__main__':
    main()
