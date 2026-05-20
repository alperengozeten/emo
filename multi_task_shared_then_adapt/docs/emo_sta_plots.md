# EMO-STA Plots

## OOD Holdout Plots: 60 / 15 / 120

### Circle Packing

![Circle Packing OOD 60/15/120](figures/circle_packing_ood_b30_by_holdout_adaptation_methods_s60_a15_b30_all_methods.png)

**Caption**

OOD holdout performance for **circle packing** at the fixed **60 / 15 / 120** budget setting. The budget is reported as **Shared / Adapt / Total**, where the total equals \(30 \times 4\) single-task iterations. Bars report **mean OOD normalized score across LLMs** as a function of held-out task size \(N\). The comparison includes **STA-Shared**, **Single-task**, **STA Warmstart**, **STA Best-Local Program**, and **STA Best-Shared Program**.

**LaTeX**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{multi_task_shared_then_adapt/figures/circle_packing_ood_b30_by_holdout_adaptation_methods_s60_a15_b30_all_methods.pdf}
    \caption{OOD holdout performance for \textit{circle packing} at the fixed \textit{60 / 15 / 120} budget setting. The budget is reported as Shared / Adapt / Total, where the total equals $30 \times 4$ single-task iterations. Bars report mean OOD normalized score across LLMs as a function of held-out task size $N$. The comparison includes \textit{STA-Shared}, \textit{Single-task}, \textit{STA Warmstart}, \textit{STA Best-Local Program}, and \textit{STA Best-Shared Program}.}
    \label{fig:cp-ood-s60-a15-b30-all-methods}
\end{figure}
```

### Circle Packing Rectangle

![Circle Packing Rectangle OOD 60/15/120](figures/circle_packing_rectangle_ood_b30_by_holdout_adaptation_methods_s60_a15_b30_all_methods.png)

**Caption**

OOD holdout performance for **circle packing rectangle** at the fixed **60 / 15 / 120** budget setting. The budget is reported as **Shared / Adapt / Total**, where the total equals \(30 \times 4\) single-task iterations. Bars report **mean OOD normalized score across LLMs** as a function of held-out task size \(N\). The comparison includes **STA-Shared**, **Single-task**, **STA Warmstart**, **STA Best-Local Program**, and **STA Best-Shared Program**.

**LaTeX**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{multi_task_shared_then_adapt/figures/circle_packing_rectangle_ood_b30_by_holdout_adaptation_methods_s60_a15_b30_all_methods.pdf}
    \caption{OOD holdout performance for \textit{circle packing rectangle} at the fixed \textit{60 / 15 / 120} budget setting. The budget is reported as Shared / Adapt / Total, where the total equals $30 \times 4$ single-task iterations. Bars report mean OOD normalized score across LLMs as a function of held-out task size $N$. The comparison includes \textit{STA-Shared}, \textit{Single-task}, \textit{STA Warmstart}, \textit{STA Best-Local Program}, and \textit{STA Best-Shared Program}.}
    \label{fig:cp-rect-ood-s60-a15-b30-all-methods}
\end{figure}
```

### Heilbronn Triangle

![Heilbronn Triangle OOD 60/15/120](figures/heilbronn_triangle_ood_b30_by_holdout_adaptation_methods_s60_a15_b30_all_methods.png)

**Caption**

OOD holdout performance for **Heilbronn triangle** at the fixed **60 / 15 / 120** budget setting. The budget is reported as **Shared / Adapt / Total**, where the total equals \(30 \times 4\) single-task iterations. Bars report **mean OOD normalized score across LLMs** as a function of held-out task size \(N\). The comparison includes **STA-Shared**, **Single-task**, **STA Warmstart**, **STA Best-Local Program**, and **STA Best-Shared Program**.

**LaTeX**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{multi_task_shared_then_adapt/figures/heilbronn_triangle_ood_b30_by_holdout_adaptation_methods_s60_a15_b30_all_methods.pdf}
    \caption{OOD holdout performance for \textit{Heilbronn triangle} at the fixed \textit{60 / 15 / 120} budget setting. The budget is reported as Shared / Adapt / Total, where the total equals $30 \times 4$ single-task iterations. Bars report mean OOD normalized score across LLMs as a function of held-out task size $N$. The comparison includes \textit{STA-Shared}, \textit{Single-task}, \textit{STA Warmstart}, \textit{STA Best-Local Program}, and \textit{STA Best-Shared Program}.}
    \label{fig:heilbronn-ood-s60-a15-b30-all-methods}
\end{figure}
```

## Best-Local Gain Profile

![EMO-STA Best-Local Gain Profile](figures/emo_sta_subtask_gain_profile_best_local.svg)

**Caption**

Subtask-level improvement profile for the **STA Best-Local** adaptation path across the selected EMO-STA table settings. Each panel corresponds to one family, and each row corresponds to one in-distribution source task. The **open marker** shows the score of the **best shared before adaptation** and the **filled marker** shows the final score after **STA Best-Local** adaptation. The x-axis reports **improvement vs single-task baseline**, so points farther to the right indicate stronger absolute performance relative to the direct single-task run.

**LaTeX**

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{multi_task_shared_then_adapt/figures/emo_sta_subtask_gain_profile_best_local.pdf}
    \caption{Subtask-level improvement profile for the \textit{STA Best-Local} adaptation path across the selected EMO-STA table settings. Each panel corresponds to one family, and each row corresponds to one in-distribution source task. The open marker shows the score of the best shared before adaptation and the filled marker shows the final score after \textit{STA Best-Local} adaptation. The x-axis reports improvement vs single-task baseline, so points farther to the right indicate stronger absolute performance relative to the direct single-task run.}
    \label{fig:emo-sta-best-local-gain-profile}
\end{figure*}
```
