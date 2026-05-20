# Representative EMO-STA Trajectory Appendix Snippet

This file contains copy-ready LaTeX for the four currently retained sample trajectory figures in `figures/sample_trajectories/`. The `Single-task` references in the adaptation panels are five-run averages over the corresponding independent single-task baseline runs.

## Result Summary

| Example | Model / method | Budget (Shared / Adapt / Total) | Shared final | Adapt start -> final | Single-task avg. |
| --- | --- | ---: | ---: | ---: | ---: |
| Circle packing | Haiku-4.5, STA Best-Local | 60/15/120 | 0.833 | 0.903 -> 0.925 | 0.865 |
| Circle packing rectangles | Sonnet-4.5, STA Best-Shared | 60/15/120 | 0.894 | 0.894 -> 0.924 | 0.840 |
| Heilbronn triangle | Sonnet-4.6, STA Best-Shared | 60/15/120 | 0.750 | 0.750 -> 0.905 | 0.678 |
| Signal processing | Opus-4.6, STA Best-Local | 60/10/100 | 0.619 | 0.635 -> 0.685 | 0.648 |

## LaTeX Code

```latex
\subsection{Representative EMO-STA Trajectories}
\label{app:representative-emosta-trajectories}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{llcccc}
\toprule
Example & Model / method & Budget & Shared & Adapt & Single-task \\
\midrule
Circle packing
& Haiku-4.5, STA Best-Local
& $60/15/120$ & $.833$ & $.903 \rightarrow .925$ & $.865$ \\
Circle packing rectangles
& Sonnet-4.5, STA Best-Shared
& $60/15/120$ & $.894$ & $.894 \rightarrow .924$ & $.840$ \\
Heilbronn triangle
& Sonnet-4.6, STA Best-Shared
& $60/15/120$ & $.750$ & $.750 \rightarrow .905$ & $.678$ \\
Signal processing
& Opus-4.6, STA Best-Local
& $60/10/100$ & $.619$ & $.635 \rightarrow .685$ & $.648$ \\
\bottomrule
\end{tabular}
\caption{\small{Representative EMO-STA trajectory examples used in the appendix. The budget column reports Shared / Adapt / Total iterations, where Total equals the number of tasks times the single-task baseline budget. The Shared column reports the final family-average score at the end of shared evolution. The Adapt column reports the average score at the start and end of task-specific adaptation. Single-task reports the mean over five independent single-task runs for the same family, model, and baseline budget.}}
\label{tab:representative-emosta-trajectories}
\end{table}

These trajectories illustrate two complementary ways in which EMO-STA improves over direct single-task search. In the circle-packing examples, shared evolution already discovers a reusable geometric solver scaffold, and task-specific adaptation mostly acts as local calibration: it improves the selected circle counts or rectangle sizes without replacing the shared representation. The Heilbronn example shows a larger adaptation effect, with every subtask improving during adaptation and the family average rising from $.750$ to $.905$, well above the five-run single-task average. The signal-processing example shows a more targeted form of adaptation: the shared program is already competitive on trend, multifrequency, and chirp signals, while adaptation mainly repairs the step-change task, raising it from $.694$ to $.883$ and moving the family average above the single-task reference. Overall, these cases suggest that EMO-STA is useful both when shared evolution finds a strong general program that needs light retuning and when adaptation must make a focused task-local correction to a shared scaffold.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/sample_trajectories/circle_packing_s60_a15_b30_haiku45_seed42_bestlocal.pdf}
    \caption{\small{Circle-packing trajectory for Haiku-4.5, seed 42, using STA Best-Local with a $60/15/120$ Shared / Adapt / Total budget. The Total budget equals $30 \times 4$ single-task iterations. The adapted family average increases from $.903$ to $.925$, compared with a five-run Single-task average of $.865$.}}
    \label{fig:traj-circle-packing-haiku45-bestlocal}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/sample_trajectories/circle_packing_rectangles_s60_a15_b30_sonnet45_seed45_bestshared.pdf}
    \caption{\small{Circle-packing-rectangles trajectory for Sonnet-4.5, seed 45, using STA Best-Shared with a $60/15/120$ Shared / Adapt / Total budget. The Total budget equals $30 \times 4$ single-task iterations. Adaptation improves the average score from $.894$ to $.924$, above the five-run Single-task average of $.840$.}}
    \label{fig:traj-circle-packing-rectangles-sonnet45-bestshared}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/sample_trajectories/heilbronn_triangle_s60_a15_b30_sonnet46_seed44_bestshared.pdf}
    \caption{\small{Heilbronn-triangle trajectory for Sonnet-4.6, seed 44, using STA Best-Shared with a $60/15/120$ Shared / Adapt / Total budget. The Total budget equals $30 \times 4$ single-task iterations. This example shows broad task-specific improvement, with the average score increasing from $.750$ to $.905$ versus a five-run Single-task average of $.678$.}}
    \label{fig:traj-heilbronn-sonnet46-bestshared}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/sample_trajectories/signal_processing_s60_a10_b25_opus46_seed42_bestlocal.pdf}
    \caption{\small{Signal-processing trajectory for Opus-4.6, seed 42, using STA Best-Local with a $60/10/100$ Shared / Adapt / Total budget. The Total budget equals $25 \times 4$ single-task iterations. Adaptation is concentrated on the step-change task, which improves from $.694$ to $.883$, raising the family average from $.635$ to $.685$ above the five-run Single-task average of $.648$.}}
    \label{fig:traj-signal-processing-opus46-bestlocal}
\end{figure}
```
