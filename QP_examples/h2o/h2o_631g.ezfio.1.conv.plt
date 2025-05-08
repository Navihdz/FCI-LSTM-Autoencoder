set term pdf
set output "h2o_631g.ezfio.1.conv.pdf"
set log x
set xlabel "Number of determinants"
set ylabel "Total Energy (a.u.)"

plot "h2o_631g.ezfio.1.conv" w lp title "E_{var} state 1", "h2o_631g.ezfio.1.conv" u 1:3 w lp title "E_{var} + PT2 state 1"

