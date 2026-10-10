"""Index actual retained ONNX nodes without selecting or publishing policies."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from servo_bam.testbench.learning_evidence import digest, require


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--common-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--revalidation-reports', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a fresh node inventory directory')
    history, protocol = load(args.history_report), load(args.protocol)
    protocol_sha = digest(args.protocol)
    executor_sha = protocol['evaluation_refresh_revision']['validator_sha256']
    require(history['passed'] and len(history['policies']) == 24,
            'The inventory requires verified full24 resume ancestry')
    require(history['active_protocol_sha256'] == protocol_sha,
            'The ancestry report uses a different active contract')
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    evidence = {str(args.history_report.resolve()): digest(args.history_report),
                str(args.protocol.resolve()): protocol_sha}
    for path, sha in {**history['evidence_sha256'], **frozen}.items():
        require(digest(path) == sha, 'Bound training evidence changed: ' + path)
    refreshed = {}
    for path in [*args.common_reports, *args.revalidation_reports]:
        report = load(path)
        require(report['passed'] and report['evaluation_protocol_sha256'] == protocol_sha
                and report.get('validation_executor_sha256', report.get('validator_sha256')) == executor_sha,
                'An inventory overlay uses a different or failed evaluator')
        evidence[str(path.resolve())] = digest(path)
        for row in report.get('policies', report.get('checkpoints', [])):
            key = row['checkpoint_sha256']
            if key in refreshed:
                require(refreshed[key]['onnx_sha256'] == row['onnx_sha256'],
                        'Conflicting exports for a saved actor')
            refreshed[key] = row
    rows = []
    for policy in history['policies']:
        for original in policy['history']:
            key = original['checkpoint_sha256']
            checkpoint = Path(original['checkpoint'])
            original_path = (Path(original['validation_report'])
                             if original['validation_report'] else None)
            original_report = load(original_path) if original_path else None
            onnx = checkpoint.parent / 'evaluations' / f"iteration{original['completed_iterations']:04d}" / 'policy.onnx'
            require(onnx.is_file(), 'A retained checkpoint has no saved ONNX: ' + str(onnx))
            onnx_sha = digest(onnx)
            if original_report:
                require(original_report['checkpoint_sha256'] == key
                        and original_report['onnx_sha256'] == onnx_sha,
                        'An original export differs from its ancestry-bound report')
            refresh = refreshed.get(key)
            validation_path = original_path
            if refresh:
                require(refresh['policy'] == policy['policy'] and refresh['seed'] == policy['seed']
                        and refresh['completed_iterations'] == original['completed_iterations']
                        and refresh['training_protocol_sha256'] == original['protocol_sha256']
                        and refresh['onnx_sha256'] == onnx_sha,
                        'A current-evaluator overlay changed the saved actor identity')
                validation_path = Path(refresh['validation_report'])
                require(digest(validation_path) == refresh['validation_report_sha256'],
                        'The current-evaluator report changed')
            validation = load(validation_path) if validation_path else None
            if validation:
                require(validation['checkpoint_sha256'] == key
                        and validation['onnx_sha256'] == onnx_sha,
                        'A validation report refers to a different actor or export')
            current = bool(validation and validation['protocol_sha256'] == protocol_sha
                           and validation['source_sha256'] == executor_sha
                           and validation['passed'] is True and validation['fault_count'] == 0)
            number = int(policy['policy'][1:])
            row = {
                'policy': policy['policy'], 'seed': policy['seed'],
                'completed_iterations': original['completed_iterations'],
                'S': f'S{number % 2}', 'O': f'O{number // 2 % 2}', 'R': f'R{number // 4}',
                'onnx': str(onnx), 'onnx_sha256': onnx_sha,
                'release_path_if_selected': f"nodes/iteration{original['completed_iterations']:04d}/{policy['policy']}-seed{policy['seed']}/policy.onnx",
                'checkpoint_sha256': key, 'training_protocol_sha256': original['protocol_sha256'],
                'original_validation_state': original['validation_state'],
                'current_executor_validation_passed': current,
                'validation_report': str(validation_path) if validation_path else '',
                'evaluation_protocol_sha256': validation['protocol_sha256'] if validation else '',
                'validation_executor_sha256': validation['source_sha256'] if validation else '',
                'current_mae_deg': validation['mae_deg'] if current else None,
                'frozen_M6_sha256': protocol['frozen_M6_sha256'],
                'frozen_O1_sha256': protocol['frozen_O1_sha256'] if number // 2 % 2 else '',
                'runtime_qualified': False, 'selected_for_release': False,
                'hardware_executed': False,
            }
            evidence[str(onnx)] = onnx_sha
            if validation_path:
                evidence[str(validation_path)] = digest(validation_path)
            rows.append(row)
    require(len(rows) == history['verified_checkpoint_count'], 'The retained node count changed')
    identities = {(r['policy'], r['seed'], r['completed_iterations']) for r in rows}
    require(len(identities) == len(rows), 'Duplicate training node identity')
    rows.sort(key=lambda r: (r['completed_iterations'], r['policy'], protocol['seeds'].index(r['seed'])))
    counts, current_counts = Counter(), Counter()
    for row in rows:
        counts[row['completed_iterations']] += 1
        if row['current_executor_validation_passed']:
            current_counts[row['completed_iterations']] += 1
    common = sorted(budget for budget, count in counts.items() if count == 24)
    current_common = sorted(budget for budget, count in current_counts.items() if count == 24)
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen contract changed during inventory: ' + path)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'source.py')
    with (args.output / 'onnx-node-index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        'passed': True, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'retained_onnx_count': len(rows), 'nodes_per_budget': dict(sorted(counts.items())),
        'current_executor_validated_nodes_per_budget': dict(sorted(current_counts.items())),
        'all24_saved_budgets': common, 'all24_current_executor_validated_budgets': current_common,
        'evaluation_protocol_sha256': protocol_sha, 'validation_executor_sha256': executor_sha,
        'evidence_sha256': evidence, 'index_sha256': digest(args.output / 'onnx-node-index.csv'),
        'source_sha256': digest(__file__), 'frozen_contract_files_unchanged': len(frozen),
        'selected_for_release': 0, 'runtime_qualification_performed': False,
        'formal_convergence_claimed': False, 'hardware_executed': False,
    }
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    text = ['# 实际保存的 ONNX 训练节点', '',
            f'共 {len(rows)} 份，逐份核对实际文件 SHA256 和保留训练来源。', '',
            '| 训练轮数 | 已保存 ONNX | 当前验证器通过 |', '| ---: | ---: | ---: |']
    text.extend(f'| {budget} | {count} | {current_counts[budget]} |' for budget, count in sorted(counts.items()))
    text.extend(['', '完整24份均已保存的节点：' + '、'.join(map(str, common)) + '。',
                 '完整24份均通过当前仿真验证的节点：' + '、'.join(map(str, current_common)) + '。', '',
                 '索引只登记实际文件；不把旧训练身份改写成新协议，不把旧验证成绩当作当前成绩。'
                 '训练节点比较应在实测前封存共同节点及匹配任务，正式方案比较使用共同最终预算。', '',
                 'ONNX 尚需对应观测器/动作配置、实机运行软件验收及最终封存。这里未发布正式策略。'
                 '选定的 ONNX 和必要部署清单进入 policy-releases/sc0090-testbench/；'
                 '全部 checkpoint、日志、评估轨迹与此动态研究索引保留在本地，不提交训练过程。', ''])
    (args.output / 'NODES.md').write_text('\n'.join(text))
    print(json.dumps({k: result[k] for k in ['passed', 'retained_onnx_count',
        'all24_current_executor_validated_budgets', 'selected_for_release']}))


if __name__ == '__main__':
    main()
