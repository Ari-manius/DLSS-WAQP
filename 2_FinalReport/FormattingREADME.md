# Formal Requirements 
- About 10 pages in total 

# Images 
- 1 Images in a row
``` Latex
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth, height=5cm, keepaspectratio]{Visual/Distributions.png}
    \caption{Total Number of Speeches by Parties and Speech Lengths for all Parties}
    \label{fig:pca_combined}
\end{figure}
```

- 2 Images next to each other 
``` Latex
\begin{figure}[H]
    \begin{subfigure}{0.45\textwidth} % Adjust width as needed
        \centering
        \includegraphics[width=\textwidth]{Visual/ModelA_Eval.png}
        \label{fig:Number of Components}
    \end{subfigure}
    \hfill % This adds space between the two subfigures
    \begin{subfigure}{0.45\textwidth} % Adjust width as needed
        \centering
        \includegraphics[width=\textwidth]{Visual/ModelB_Eval.png}
        \label{fig:Biggest (weakly) Connected Component}
    \end{subfigure}
    \caption{Evaltuations of Base and Tuned Models for Task A and B}
    \label{fig:pca_combined}
\end{figure}
```

- 3 three images in a row 
``` Latex
\begin{figure}[H]
    \begin{subfigure}{0.3\textwidth} 
        \centering
        \includegraphics[width=\textwidth]{Visual/wordclouds_per_party/wordcloud_NEOS.png} %Path to images 
        \label{fig:Multilayer Perceptron}
    \end{subfigure}
    \hfill 
    \begin{subfigure}{0.3\textwidth} 
        \centering
        \includegraphics[width=\textwidth]{Visual/wordclouds_per_party/wordcloud_ÖVP.png}
        \label{fig:Concolutional Graph Neural Network}
    \end{subfigure}
    \hfill 
    \begin{subfigure}{0.3\textwidth} 
        \centering
        \includegraphics[width=\textwidth]{Visual/wordclouds_per_party/wordcloud_SPÖ.png}
        \label{fig:GraphSAGE Model} %Invisible! 
    \end{subfigure}
    \caption{Wordclouds for Speeches by Parties II} %Visible Caption 
    \label{fig:}
\end{figure}
```

# References
- Copy the references and citations into the references.bib file in the bibtex format
- Quote in the text by `[-@citation]` or `[-@citation1 ; -@citation1]`