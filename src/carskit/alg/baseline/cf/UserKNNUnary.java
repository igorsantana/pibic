package carskit.alg.baseline.cf;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import carskit.generic.Recommender;
import happy.coding.io.Lists;
import happy.coding.io.Strings;
import librec.data.DenseVector;
import librec.data.SparseVector;
import librec.data.SymmMatrix;

public class UserKNNUnary extends Recommender {
    // user: nearest neighborhood
    private SymmMatrix userCorrs;
    private DenseVector userMeans;


    public UserKNNUnary(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "UserKNN";

    }


    @Override
    protected void initModel() throws Exception {
        super.initModel();
        userCorrs = buildCorrs(true);
        userMeans = new DenseVector(numUsers);
        for (int u = 0; u < numUsers; u++) {
            SparseVector uv = train.row(u);
            userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
        }
    }
    
    @Override
    protected double predict(int u, int j, int c, Set<Integer> obs) throws Exception {
        if(isUserSplitting)
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
        if(isItemSplitting)
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;

        return predict(u,j,obs);
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {

        if(isUserSplitting)
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
        if(isItemSplitting)
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;

            return predict(u,j);
    }

    @Override
    protected double predict(int u, int j, Set<Integer> obs) throws Exception {
        // find a number of similar users
        Map<Integer, Double> nns = new HashMap<>();

        SparseVector dv = userCorrs.row(u);
        for (int v : dv.getIndex()) {
            double sim = dv.get(v);
            double rate = train.get(v, j);

            if (isRankingPred && rate > 0)
                nns.put(v, sim); // similarity could be negative for item ranking
            else if (sim > 0 && rate > 0)
                nns.put(v, sim);
        }

        // topN similar users
        if (knn > 0 && knn < nns.size()) {
            List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
            List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
            nns.clear();
            for (Map.Entry<Integer, Double> kv : subset)
                nns.put(kv.getKey(), kv.getValue());
        }

        if (nns.size() == 0)
            return  globalMean;
        else {
            double sum = 0, ws = 0;
            HashSet<Integer> obsv = new HashSet<>();
            
            for (int o : obs)
            	obsv.add(o);
            
            for (Entry<Integer, Double> en : nns.entrySet()) {
                int v = en.getKey();
                double sim = en.getValue();
                
                if (obsv.contains(v) )
                	sum += sim;
                
                ws += Math.abs(sim);
            }

            return ws > 0 ? sum / ws : globalMean;
        }

    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
    }

}
