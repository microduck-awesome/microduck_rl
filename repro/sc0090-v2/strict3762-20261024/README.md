# 3762仅修正碰撞的历史重训记录

训练及最终评测尚未完成。入口：

- [冻结的单GPU历史协议](STRICT_3762_SINGLE_GPU.md)
- [外部执行器审阅](EXECUTION_REVIEW.md)
- [polish单向保持语义与2500次行为退化](POLISH_STATE_REVIEW_ADDENDUM.md)
- [训练后产物与行为评测审阅](FINAL_EVALUATION_REVIEW.md)

formal-evidence保留实际分段、恢复和评测证据。配置/进程检查PASS只证明执行契约，不等同于行为验收通过。已知2500次右侧卧退化记录必须保留在最终报告中，不以polish标志代替实际成功率。

补充：2750次原复评已恢复1792/1792，未回退或修改配方；2500次失败仍保留，并安排三检查点轨迹核验。
