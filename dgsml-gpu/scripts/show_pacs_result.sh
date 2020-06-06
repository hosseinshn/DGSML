cd ../pacs_res
for R in 0.95 0.8 0.5 0.2; do 
    for T in art_painting cartoon photo sketch; do
        echo $R;
        echo $T;
        for N in {1..5}; do
            F="save${R}_${N}_${T}/Target.txt"
            if ls $F 1> /dev/null 2>&1; then
                cat $F | tail -1 | cut -d':' -f2;
            else
                echo "NA"
            fi

        done;

    done;
done | paste -d' ' - - - - - - -;

cd -

