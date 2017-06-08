// Copyright (C) 2016 Raoni Ferreira
//
// This file is part of CARSKit.
//
// CARSKit is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CARSKit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CARSKit. If not, see <http://www.gnu.org/licenses/>.
//

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
import librec.util.Logs;

/**
 * Domingues, M. A., et al. "Dimensions as Virtual Items: Improving the predictive ability of top-N recommender systems." Information Processing and Management (2012).<p></p>
 * Note: This implementation is modified version from original CARSKit ItemKNN.
 *       The original ItemKNN score function which regard explicit data (ie, ratings) 
 *       was replace by a version which regard implicit data and virtual items.
 *
 * @author Raoni Ferreira
 *
 */

public class ItemKNNUnary extends Recommender {

    // user: nearest neighborhood
    private SymmMatrix itemCorrs;
    private DenseVector itemMeans;

	public ItemKNNUnary(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {
		// TODO Auto-generated constructor stub
        super(trainMatrix, testMatrix, fold);
        this.algoName = "ItemKNNUnary";

	}

    @Override
    protected void initModel() throws Exception {
        super.initModel();
        
        itemCorrs = buildCorrs(false);
        itemMeans = new DenseVector(numItems);
        for (int i = 0; i < numItems; i++) {
            SparseVector vs = train.column(i);
            itemMeans.set(i, vs.getCount() > 0 ? vs.mean() : globalMean);
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
    protected double predict(int u, int j, Set<Integer> obs) throws Exception{

        // find a number of similar items
        Map<Integer, Double> nns = new HashMap<>();

        SparseVector dv = itemCorrs.row(j);
        for (int i : dv.getIndex()) {
            double sim = dv.get(i);
            double rate = train.get(u, i);
            if (sim >0 && rate > 0)
            	nns.put(i, sim);
            //if (isRankingPred && rate > 0)
            //    nns.put(i, sim);
            //else if (sim > 0 && rate > 0)
            //    nns.put(i, sim);
        }

        // topN similar items
        if (knn > 0 && knn < nns.size()) {
            List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
            List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
            nns.clear();
            for (Map.Entry<Integer, Double> kv : subset)
                nns.put(kv.getKey(), kv.getValue());
        }

        if (nns.size() == 0)
            return globalMean;
        else {
            double sum = 0, ws = 0;
            HashSet<Integer> obsv = new HashSet<>();
            
            for (int o : obs)
            	obsv.add(o);
            
            for (Entry<Integer, Double> en : nns.entrySet()) {
                int i = en.getKey();
                double sim = en.getValue();                
                
                if (obsv.contains(i) )
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
