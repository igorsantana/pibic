// Copyright (C) 2015 Yong Zheng
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

package carskit.generic;


import com.google.common.collect.*;
import com.google.common.primitives.Doubles;
import happy.coding.io.FileConfiger;
import happy.coding.io.FileIO;
import happy.coding.io.LineConfiger;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.math.Measures;
import happy.coding.math.Randoms;
import happy.coding.math.Sims;
import happy.coding.math.Stats;
import happy.coding.system.Dates;
import happy.coding.system.Debug;

import java.text.DecimalFormat;
import java.util.*;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import carskit.data.processor.DataDAO;
import carskit.data.structure.SparseMatrix;
import carskit.data.setting.*;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.data.MatrixEntry;

import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.cache.LoadingCache;


public abstract class Recommender implements Runnable{

    // default temporary file directory
    public static String workingPath;
    /************************************ Static parameters for all recommenders ***********************************/
    // configer
    public static FileConfiger cf;
    // matrix of rating data
    public static SparseMatrix rateMatrix;

    // params used for multiple runs
    public static Map<String, List<Float>> params = new HashMap<>();

    // Guava cache configuration
    protected static String cacheSpec;

    // number of cpu cores used for parallelization
    protected static int numCPUs;

    // verbose
    protected static boolean verbose = true;

    // line configer for item ranking, evaluation
    protected static LineConfiger rankOptions, algoOptions;

    // is ranking/rating prediction
    public static boolean isRankingPred;
    // threshold to binarize ratings
    public static float binThold;
    // the ratio of validation data split from training data
    public static float validationRatio;
    // is diversity-based measures used
    protected static boolean isDiverseUsed;
    // early-stop criteria
    protected static Measure earlyStopMeasure = null;
    // is save model
    protected static boolean isSaveModel = false;
    // view of rating predictions
    public static String view;

    // rate DAO object
    public static DataDAO rateDao;


    // number of recommended items
    protected static int numRecs, numIgnore;

    // a list of rating scales
    protected static List<Double> ratingScale;
    // number of rating levels
    protected static int numLevels;
    // Maximum, minimum values of rating scales
    protected static double maxRate, minRate;

    // init mean and standard deviation
    protected static double initMean, initStd;
    // small value for initialization
    protected static double smallValue = 0.01;

    // number of nearest neighbors
    protected static int knn;
    // similarity measure
    protected static String similarityMeasure;
    // number of shrinkage
    protected static int similarityShrinkage;

    /**
     * An indicator of initialization of static fields. This enables us to control when static fields are initialized,
     * while "static block" will be always initialized or executed. The latter could cause unexpected exceptions when
     * multiple runs (with different configuration files) are conducted sequentially, because some static settings will
     * not be override in such a "staic block".
     */
    public static boolean resetStatics = true;

    /************************************ Recommender-specific parameters ****************************************/
    // algorithm's name
    public String algoName;
    // current fold
    protected int fold;
    // fold information
    protected String foldInfo;
    // is output recommendation results
    protected boolean isResultsOut = true;

    // number of users, items, ratings
    protected int numUsers, numItems, numRates;

    // user-vector cache, item-vector cache
    protected LoadingCache<Integer, SparseVector> userCache, itemCache;

    // user-items cache, item-users cache
    protected LoadingCache<Integer, List<Integer>> userItemsCache, itemUsersCache;

    // rating matrix for training, validation and test
    protected SparseMatrix trainMatrix, validationMatrix, testMatrix;
    protected boolean isUserSplitting=false, isItemSplitting=false, isCARSRecommender=false;
    
    protected Table<Integer, Integer, Integer> userIdMapper, itemIdMapper;
    
    protected boolean isDaVIBest=false, isDaVIAll=false; //inclusion of DaVI approach
    
    protected int daviBestDim=-1;
    
    protected HashMap<Integer, String> itemVirtualIds, userVirtualIds;
    
    protected HashMap<Integer, String> userVirtualIdsTrain, itemVirtualIdsTrain;
    
    // variables used in itemFrequency
    protected HashMap<Integer, Integer> itemFrequency;
    protected Integer freqNumItems;
    protected boolean shouldUseItemFrequency;
    //protected HashMap<Integer, String> userVirtualIdsTest, itemVirtualIdsTest;
    
    protected BiMap<Integer, Integer> userMapper, itemMapper, ctxMapper, testDaviUserMapper, testDaviItemMapper;
    protected HashMap<Integer, Integer> userUIMapperTrain, itemUIMapperTrain, userUIMapperTest, itemUIMapperTest;
    protected Collection<Integer> conditions;

    protected librec.data.SparseMatrix train; // this is traditional 2D rating matrix, transformed from either original data, or the splitting process

    // upper symmetric matrix of item-item correlations
    protected SymmMatrix corrs;

    // performance measures
    public Map<Measure, Double> measures;
    // global average of training rates
    protected double globalMean;

    /**
     * Recommendation measures
     *
     */
    public enum Measure {
        /* prediction-based measures */
        MAE, RMSE, NMAE, rMAE, rRMSE, MPE, Perplexity,
        /* ranking-based measures */
        D5, D10, Pre1, Pre3, Pre5, Pre10, Rec1, Rec3, Rec5, Rec10, MAP, MRR, NDCG, AUC, F11, F13, F15, F110,
        MAP1, MAP3, MAP5, MAP10,
        /* execution time */
        TrainTime, TestTime,
        /* loss value */
        Loss
    }


    public Recommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {

        // config recommender
        if (cf == null || rateMatrix == null) {
            Logs.error("Recommender is not well configed");
            System.exit(-1);
        }

        numUsers = rateDao.numUsers();
        numItems = rateDao.numItems();

        // static initialization (reset), only done once
        if (resetStatics) {
            // change the indicator
            resetStatics = false;

            ratingScale = rateDao.getRatingScale();
            minRate = ratingScale.get(0);
            maxRate = ratingScale.get(ratingScale.size() - 1);
            numLevels = ratingScale.size();

            //numUsers = rateDao.numUsers();
            //numItems = rateDao.numItems();

            initMean = 0.0;
            initStd = 0.1;

            cacheSpec = cf.getString("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");

            rankOptions = cf.getParamOptions("item.ranking");
            isRankingPred = rankOptions.isMainOn();
            isDiverseUsed = rankOptions.contains("-diverse");
            numRecs = rankOptions.getInt("-topN", -1);
            numIgnore = rankOptions.getInt("-ignore", -1);

            LineConfiger evalOptions = cf.getParamOptions("evaluation.setup");
            view = evalOptions.getString("--test-view", "all");

            String earlyStop = evalOptions.getString("--early-stop");
            if (earlyStop != null) {
                for (Measure m : Measure.values()) {
                    if (m.name().equalsIgnoreCase(earlyStop)) {
                        earlyStopMeasure = m;
                        break;
                    }
                }
            }

            int numProcessors = Runtime.getRuntime().availableProcessors();
            numCPUs = evalOptions.getInt("-cpu", numProcessors);
            Randoms.seed(evalOptions.getLong("--rand-seed", System.currentTimeMillis())); // initial random seed

            // output options
            LineConfiger outputOptions = cf.getParamOptions("output.setup");
            if (outputOptions != null) {
                verbose = outputOptions.isOn("-verbose", true);
                isSaveModel = outputOptions.contains("--save-model");
            }

            knn = cf.getInt("num.neighbors", 20);
            similarityMeasure = cf.getString("similarity", "PCC");
            similarityShrinkage = cf.getInt("num.shrinkage", 30);
            
            this.setItemFrequency(false);
        }


        this.trainMatrix = trainMatrix;
        this.testMatrix = testMatrix;

        // fold info
        this.fold = fold;
        foldInfo = fold > 0 ? " fold [" + fold + "]" : "";

        // whether to write out results
        LineConfiger outputOptions = cf.getParamOptions("output.setup");
        if (outputOptions != null) {
            //isResultsOut = outputOptions.isMainOn();
            isResultsOut = true;
        }

        // global mean
        globalMean=trainMatrix.getGlobalAvg();

        // class name as the default algorithm name
        algoName = this.getClass().getSimpleName();
        // get parameters of an algorithm
        algoOptions = getModelParams(algoName);

        // compute item-item correlations
        if (isRankingPred && isDiverseUsed)
            corrs = new SymmMatrix(numItems);
        
    }
    
    private void computeItemFrequency(){
    	this.itemFrequency = new HashMap<>();
    	this.freqNumItems = 0;
    	for(MatrixEntry me : this.trainMatrix){
    		int uiid = me.row();
    		int iid = rateDao.getItemIdFromUI(uiid);
    		if(!this.itemFrequency.containsKey(iid)){
    			this.itemFrequency.put(iid, 0);
    			
    		}
    		this.freqNumItems++;
    		this.itemFrequency.replace(iid, this.itemFrequency.get(iid) + 1);
    	}
    	
    }
    
    public void setItemFrequency(boolean value){
    	this.shouldUseItemFrequency = value;
    	if(value == true){
    		this.computeItemFrequency();
    	}
    }

    public HashMap<Integer, Integer> getUserUIMapperTrain()
    {
    	return this.userUIMapperTrain;
    }

    
    public HashMap<Integer, Integer> getUserUIMapperTest()
    {
    	return this.userUIMapperTest;
    }

    public HashMap<Integer, Integer> getItemUIMapperTrain()
    {
    	return this.itemUIMapperTrain;
    }

    
    public HashMap<Integer, Integer> getItemUIMapperTest()
    {
    	return this.itemUIMapperTest;
    }
 
    
    public void setResultsOut(boolean value)
    {
    	isResultsOut = value;
    }

    protected LineConfiger getModelParams(String algoName) {
        return cf.contains(algoName) ? cf.getParamOptions(algoName) : null;
    }

	public double recommend(int u, int j, int c) throws Exception{
		if (this.algoName.matches("ItemKNNUnary")){
			HashMap<Integer, HashMultimap<Integer, Integer>> uciList_train=rateDao.getUserCtxList(trainMatrix);
			HashMultimap<Integer, Integer> cList_train = (uciList_train.containsKey(u))?uciList_train.get(u):HashMultimap.<Integer, Integer>create();
            Set<Integer> ratedItems = (cList_train.containsKey(c))?cList_train.get(c):new HashSet<Integer>();
			return ranking(u, j, c, ratedItems);
		}
				
		return ranking(u, j, c);
	}

    protected double predict(int u, int j, int c) throws Exception {
        return globalMean;
    }

    protected double predict(int u, int j) throws Exception {
        return globalMean;
    }

    protected double predict(int u, int j, int c, Set<Integer> obs) throws Exception {
    	return globalMean;
    }

    
    protected double predict(int u, int j, Set<Integer> obs) throws Exception {
    	return globalMean;
    }

    /**
     * predict a specific rating for user u on item j. It is useful for evalution which requires predictions are
     * bounded.
     *
     * @param u
     *            user id
     * @param j
     *            item id
     *
     * @param c
     *            context id
     * @param bound
     *            whether to bound the prediction
     * @return prediction
     */

    protected double predict(int u, int j, int c, boolean bound) throws Exception {
        double pred = predict(u, j,c);

        if (bound) {
            if (pred > maxRate)
                pred = maxRate;
            if (pred < minRate)
                pred = minRate;
        }

        return pred;
    }
    
    protected double predict(int u, int j, int c, Set<Integer> obs, boolean bound) throws Exception {
        double pred = predict(u, j,c, obs);

        if (bound) {
            if (pred > maxRate)
                pred = maxRate;
            if (pred < minRate)
                pred = minRate;
        }

        return pred;
    }

    public void execute() throws Exception {

        Stopwatch sw = Stopwatch.createStarted();
        if (Debug.ON) {
            // learn a recommender model
            initModel();

            // show algorithm's configuration
            // printAlgoConfig();

            // build the model
            buildModel();

            // post-processing after building a model, e.g., release intermediate memory to avoid memory leak
            postModel();
        } else {
            /**
             * load a learned model: this code will not be executed unless "Debug.OFF" mainly for the purpose of
             * exemplifying how to use the saved models
             */
            loadModel();
        }
        long trainTime = sw.elapsed(TimeUnit.MILLISECONDS);

        // evaluation
        if (verbose)
            Logs.debug("{}{} evaluate test data ... ", algoName, foldInfo);
        
        
        
        if (isDaVIBest || isDaVIAll)
        	measures = isRankingPred ? evalRankings(getConditions()) : evalRatings();
        else
        	measures = isRankingPred ? evalRankings() : evalRatings();
        //measures = isRankingPred ? evalRankings() : evalRatings();
        	
        
        	
        String measurements = getEvalInfo(measures);
        sw.stop();
        long testTime = sw.elapsed(TimeUnit.MILLISECONDS) - trainTime;

        // collecting results
        measures.put(Measure.TrainTime, (double) trainTime);
        measures.put(Measure.TestTime, (double) testTime);

        String evalInfo = algoName + foldInfo + ": " + measurements + "\tTime: "
                + Dates.parse(measures.get(Measure.TrainTime).longValue()) + ", "
                + Dates.parse(measures.get(Measure.TestTime).longValue());
        if (!isRankingPred)
            evalInfo += "\tView: " + view;

        if (fold > 0)
            Logs.debug(evalInfo);

        if (isSaveModel)
            saveModel();
    }



    protected double correlation(SparseVector iv, SparseVector jv) {
        return correlation(iv, jv, similarityMeasure);
    }

    /**
     * Compute the correlation between two vectors for a specific method
     *
     * @param iv
     *            vector i
     * @param jv
     *            vector j
     * @param method
     *            similarity method
     * @return the correlation between vectors i and j; return NaN if the correlation is not computable.
     */
    protected double correlation(SparseVector iv, SparseVector jv, String method) {

        // compute similarity
        List<Double> is = new ArrayList<>();
        List<Double> js = new ArrayList<>();

        for (Integer idx : jv.getIndex()) {
            if (iv.contains(idx)) {
                is.add(iv.get(idx));
                js.add(jv.get(idx));
            }
        }

        double sim = 0;
        switch (method.toLowerCase()) {
            case "cos":
                // for ratings along the overlappings
                sim = Sims.cos(is, js);
                break;
            case "cos-binary":
                // for ratings along all the vectors (including one-sided 0s)
                sim = iv.inner(jv) / (Math.sqrt(iv.inner(iv)) * Math.sqrt(jv.inner(jv)));
                break;
            case "msd":
                sim = Sims.msd(is, js);
                break;
            case "cpc":
                sim = Sims.cpc(is, js, (minRate + maxRate) / 2.0);
                break;
            case "exjaccard":
                sim = Sims.exJaccard(is, js);
                break;
            case "pcc":
            default:
                sim = Sims.pcc(is, js);
                break;
        }

        // shrink to account for vector size
        if (!Double.isNaN(sim)) {
            int n = is.size();
            int shrinkage = cf.getInt("num.shrinkage");
            if (shrinkage > 0)
                sim *= n / (n + shrinkage + 0.0);
        }

        return sim;
    }

    /**
     * @return the evaluation information of a recommend
     */
    public static String getEvalInfo(Map<Measure, Double> measures) {
        String evalInfo = null;
        if (isRankingPred) {
            if (isDiverseUsed)
                evalInfo = String.format("Pre1: %.6f, Pre3: %.6f, Pre5: %.6f, Pre10: %.6f, Rec1: %.6f, Rec3: %.6f, Rec5: %.6f, Rec10: %.6f, F11: %.6f, F13: %.6f, F15: %.6f, F110: %.6f, AUC: %.6f, MAP1: %.6f, MAP3: %.6f, MAP5: %6f, MAP10: %6f, NDCG: %.6f, MRR: %.6f, D5: %.6f, D10: %.6f",
                		measures.get(Measure.Pre1), measures.get(Measure.Pre3), measures.get(Measure.Pre5), measures.get(Measure.Pre10), 
                		measures.get(Measure.Rec1), measures.get(Measure.Rec3), measures.get(Measure.Rec5), measures.get(Measure.Rec10),
                		measures.get(Measure.F11), measures.get(Measure.F13), measures.get(Measure.F15), measures.get(Measure.F110), 
                		measures.get(Measure.AUC), measures.get(Measure.MAP1), measures.get(Measure.MAP3), measures.get(Measure.MAP5), measures.get(Measure.MAP10), measures.get(Measure.NDCG), measures.get(Measure.MRR), measures.get(Measure.D5),
                        measures.get(Measure.D10));
            else
                evalInfo = String.format("Pre1: %.6f, Pre3: %.6f, Pre5: %.6f, Pre10: %.6f, Rec1: %.6f, Rec3: %.6f, Rec5: %.6f, Rec10: %.6f, F11: %.6f, F13: %.6f, F15: %.6f, F110: %.6f, AUC: %.6f, MAP1: %.6f, MAP3: %.6f, MAP5: %6f, MAP10: %6f, NDCG: %.6f, MRR: %.6f",
                   		measures.get(Measure.Pre1), measures.get(Measure.Pre3), measures.get(Measure.Pre5), measures.get(Measure.Pre10), 
                		measures.get(Measure.Rec1), measures.get(Measure.Rec3), measures.get(Measure.Rec5), measures.get(Measure.Rec10),
                		measures.get(Measure.F11), measures.get(Measure.F13), measures.get(Measure.F15), measures.get(Measure.F110), 
                        measures.get(Measure.AUC), measures.get(Measure.MAP1), measures.get(Measure.MAP3), measures.get(Measure.MAP5), measures.get(Measure.MAP10), measures.get(Measure.NDCG),
                        measures.get(Measure.MRR));

        } else {
            evalInfo = String.format("MAE: %.6f, RMSE: %.6f, NAME: %.6f, rMAE: %.6f, rRMSE: %.6f, MPE: %.6f", measures.get(Measure.MAE),
                    measures.get(Measure.RMSE), measures.get(Measure.NMAE), measures.get(Measure.rMAE),
                    measures.get(Measure.rRMSE), measures.get(Measure.MPE));

            // for some graphic models
            if (measures.containsKey(Measure.Perplexity)) {
                evalInfo += String.format(",%.6f", measures.get(Measure.Perplexity));
            }
        }

        return evalInfo;
    }

    /**
     * @return the evaluation results of rating predictions
     */
    protected Map<Measure, Double> evalRatings() throws Exception {

        List<String> preds = null;
        String toFile = null;
        if (isResultsOut) {
            preds = new ArrayList<String>(1500);
            preds.add("userId\titemId\tcontexts\trating\tprediction"); // optional: file header
            toFile = workingPath + algoName + "-rating-predictions" + foldInfo + ".txt"; // the output-file name
            FileIO.deleteFile(toFile); // delete possibly old files
        }

        double sum_maes = 0, sum_mses = 0, sum_r_maes = 0, sum_r_rmses = 0, sum_perps = 0;
        int numCount = 0, numPEs = 0;

        for (MatrixEntry me : testMatrix) {
            double rate = me.get();

            int ui = me.row();
            int ctx = me.column();
            int u=rateDao.getUserIdFromUI(ui);
            int j=rateDao.getItemIdFromUI(ui);

            if(isUserSplitting)
                u = userIdMapper.contains(u,ctx) ? userIdMapper.get(u,ctx) : u;
            if(isItemSplitting)
                j = itemIdMapper.contains(j,ctx) ? itemIdMapper.get(j,ctx) : j;

            double pred = predict(u,j, ctx, true);
            
            if(this.shouldUseItemFrequency){
				
				int freq = this.itemFrequency.get(j);
				double oldPred = pred;
				pred = pred * (1 + (freq / (this.freqNumItems - freq)));
				Logs.info("Old pred: {}, New pred: {}", oldPred, pred);
			}
            
            
            if (Double.isNaN(pred))
                continue;

            // perplexity: for some graphic model
            double perp = perplexity(u, j, pred);
            sum_perps += perp;

            // rounding prediction to the closest rating level
            double rPred = Math.round(pred / minRate) * minRate;

            double err = Math.abs(rate - pred); // absolute predictive error
            double r_err = Math.abs(rate - rPred);

            sum_maes += err;
            sum_mses += err * err;

            sum_r_maes += r_err;
            sum_r_rmses += r_err * r_err;

            numCount++;

            // output predictions
            if (isResultsOut) {
                // restore back to the original user/item id
                preds.add(rateDao.getUserId(u) + "\t" + rateDao.getItemId(j) + "\t" + rateDao.getContextSituationFromInnerId(ctx) + "\t" + rate + "\t" + (float) pred);
                if (preds.size() >= 1000) {
                    FileIO.writeList(toFile, preds, true);
                    preds.clear();
                }
            }
        }

        if (isResultsOut && preds.size() > 0) {
            FileIO.writeList(toFile, preds, true);
            Logs.debug("{}{} has writeen rating predictions to {}", algoName, foldInfo, toFile);
        }

        double mae = sum_maes / numCount;
        double rmse = Math.sqrt(sum_mses / numCount);

        double r_mae = sum_r_maes / numCount;
        double r_rmse = Math.sqrt(sum_r_rmses / numCount);

        Map<Measure, Double> measures = new HashMap<>();
        measures.put(Measure.MAE, mae);
        // normalized MAE: useful for direct comparison among different data sets with distinct rating scales
        measures.put(Measure.NMAE, mae / (maxRate - minRate));
        measures.put(Measure.RMSE, rmse);

        // MAE and RMSE after rounding predictions to the closest rating levels
        measures.put(Measure.rMAE, r_mae);
        measures.put(Measure.rRMSE, r_rmse);

        // measure zero-one loss
        measures.put(Measure.MPE, (numPEs + 0.0) / numCount);

        // perplexity
        if (sum_perps > 0) {
            measures.put(Measure.Perplexity, Math.exp(sum_perps / numCount));
        }

        return measures;
    }

    public void setIdMappers(Table<Integer, Integer, Integer> uid, Table<Integer, Integer, Integer> iid)
    {
        this.userIdMapper = uid;
        this.itemIdMapper = iid;

        if(uid!=null) isUserSplitting=true;
        if(iid!=null) isItemSplitting=true;

        this.algoName=this.getAlgoName();
    }

    
    public void setConditions(Collection<Integer> conditions)
    {	
    	this.conditions = conditions;
    }
    
    public Collection<Integer> getConditions()
    { return this.conditions; }
    
    
    public void setDaviBestDimension(int dimension) {
    	if (isDaVIBest)
    		this.daviBestDim = dimension;
    }
    
    public String getDaviBestDimension() {
    	if (isDaVIBest && this.daviBestDim != -1)
    		return rateDao.getContextDimensionId(this.daviBestDim);
    	else
    		return "none";
    }
    
    
    public void setDaVIMappers(String daviApproach, HashMap<Integer, Integer> userUIMapperTrain, HashMap<Integer, Integer> itemUIMapperTrain,
    							HashMap<Integer, Integer> userUIMapperTest, HashMap<Integer, Integer> itemUIMapperTest)
    {
    	
    	if(daviApproach.matches("davibest") && userUIMapperTrain!=null && itemUIMapperTrain!=null && userUIMapperTest!=null && itemUIMapperTest!=null) isDaVIBest=true;
    	
    	if(daviApproach.matches("daviall") && userUIMapperTrain!=null && itemUIMapperTrain!=null && userUIMapperTest!=null && itemUIMapperTest!=null) isDaVIAll=true;
    	
    	this.userUIMapperTrain = userUIMapperTrain;
    	this.itemUIMapperTrain = itemUIMapperTrain;
    	this.userUIMapperTest = userUIMapperTest;
    	this.itemUIMapperTest = itemUIMapperTest;
    	
    	this.algoName=this.getAlgoName();
    	
    }

    public void setVirtualItems(HashMap<Integer, String> itemVirtualIdsTrain)
    {
    	if(itemVirtualIdsTrain!=null) isDaVIBest=true;
    	this.itemVirtualIdsTrain = itemVirtualIdsTrain;
    	this.numItems = this.numItems + this.itemVirtualIdsTrain.size();
    }


    protected String getAlgoName()
    {
        if(isUserSplitting && isItemSplitting)
            this.algoName = "UISplitting-"+this.algoName;
        else
        {
            if(isUserSplitting==true)
                this.algoName = "UserSplitting-"+this.algoName;
            else if(isItemSplitting==true)
                this.algoName = "ItemSplitting-"+this.algoName;
            else if (isDaVIBest==true)
            	this.algoName = "DaVIBest-"+this.algoName;
            else if (isDaVIAll==true)
            	this.algoName = "DaVIAll-"+this.algoName;
        }
        return this.algoName;
    }
    
    protected librec.data.SparseMatrix createTraditionalSparseMatrixFromDaVI(SparseMatrix sm, HashMap<Integer, Integer> userUIMapper, HashMap<Integer, Integer> itemUIMapper)
    {
        Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        for(int uiid:sm.rows()){
            int uid=userUIMapper.get(uiid);
            int iid=itemUIMapper.get(uiid);
            SparseVector sv=sm.row(uiid);
            if(sv.getCount()>0) {
                dataTable.put(uid, iid, sv.mean());
                colMap.put(iid,uid);
            }
        }

        
        return new librec.data.SparseMatrix(this.numUsers, this.numItems, dataTable, colMap);
    }

    protected librec.data.SparseMatrix createTraditionalSparseMatrixBySplitting()
    {
        Table<Integer, Integer, Double> newtable = HashBasedTable.create();
        Multimap<Integer, Integer> colMap = HashMultimap.create();
        HashMultimap<String, Double> records=HashMultimap.create();
        int maxUser=-1, maxItem=-1;

        for(MatrixEntry me:this.trainMatrix){
            int ui=me.row();
            int ctx=me.column();
            int u=rateDao.getUserIdFromUI(ui);
            int j=rateDao.getItemIdFromUI(ui);
            

            if(isUserSplitting)
                u = userIdMapper.contains(u,ctx) ? userIdMapper.get(u,ctx) : u;
            if(isItemSplitting)
                j = itemIdMapper.contains(j,ctx) ? itemIdMapper.get(j,ctx) : j;
            records.put(u + "," + j, me.get());
            if(u>maxUser) maxUser=u;
            if(j>maxItem) maxItem=j;
        }

        for(String key:records.keySet()){
            String[] ids=key.split(",");
            int u=Integer.valueOf(ids[0]);
            int j=Integer.valueOf(ids[1]);
            newtable.put(u, j, Stats.mean(records.get(key)));
            colMap.put(j, u);
        }
        maxUser++;
        maxItem++;
        this.numUsers= (numUsers<maxUser)? maxUser: numUsers;
        this.numItems= (numItems<maxItem)? maxItem: numItems;

        librec.data.SparseMatrix sm=new librec.data.SparseMatrix(numUsers, numItems, newtable, colMap);
        Logs.info("Density of transformed 2D rating matrix ============================== "+(sm.getData().length+0.0)/(sm.numRows()*sm.numColumns()));
        return sm;
    }

//    protected librec.data.SparseMatrix createTraditionalSparseMatrixFromDaVI()
//    {
//        Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
//        Multimap<Integer, Integer> colMap = HashMultimap.create();
//        int numVirtualItems=0;
//        
//        for(int uiid:trainMatrix.rows()){
//        	int uid=rateDao.getUserIdFromUI(uiid);
//        	int iid=rateDao.getItemIdFromUI(uiid);
//        	SparseVector sv=train.row(uiid);
//        	if(sv.getCount()>0) {
//        		dataTable.put(uid, iid, sv.mean());
//        		colMap.put(iid, uid);
//        	}
//        }
//
//        return new librec.data.SparseMatrix(rateDao.numUsers(), rateDao.numItems()+numVirtualItems, dataTable, colMap);
//    }
    
    protected librec.data.SparseMatrix createTraditionalSparseMatrixByDaVI()
    {
        Table<Integer, Integer, Double> newtable = HashBasedTable.create();
        Multimap<Integer, Integer> colMap = HashMultimap.create();
        HashMultimap<String, Double> records=HashMultimap.create();
        int maxUser=-1, maxItem=-1;
        
        for(MatrixEntry me:this.trainMatrix){
            int ui=me.row();
            int ctx=me.column();
            int u=rateDao.getUserIdFromUI(ui);
            int j=rateDao.getItemIdFromUI(ui);

            records.put(u + "," + j, me.get());
            records.put(u + "," + ctx, me.get());
            
            if(u>maxUser) maxUser=u;
            if(j>maxItem) maxItem=j;
            if(ctx>maxItem) maxItem=ctx;
        }

        for(String key:records.keySet()){
            String[] ids=key.split(",");
            int u=Integer.valueOf(ids[0]);
            int j=Integer.valueOf(ids[1]);
            newtable.put(u, j, Stats.mean(records.get(key)));
            colMap.put(j, u);
        }
        maxUser++;
        maxItem++;
        this.numUsers= (numUsers<maxUser)? maxUser: numUsers;
        this.numItems= (numItems<maxItem)? maxItem: numItems;

        //Logs.debug("Fold["+fold+"]: numUsers = "+numUsers + ", numItems = "+numItems);
        librec.data.SparseMatrix sm=new librec.data.SparseMatrix(numUsers, numItems, newtable, colMap);
        Logs.info("Density of transformed 2D rating matrix ============================== "+(sm.getData().length+0.0)/(sm.numRows()*sm.numColumns()));
        return sm;
    }

    

    protected double perplexity(int u, int j, double r) throws Exception {
        return 0;
    }
    
    /** this evalRanking method is only called when the DaVIBest algorithm is used
     *  the other algorithms use evalRanking implemented just below.
     * @return the evaluation results of ranking predictions
     */
    
    protected Map<Measure, Double> evalRankings(Collection <Integer> conditions) throws Exception {
    	
    	HashMap<Integer, HashMultimap<Integer, Integer>> uciList=rateDao.getUserCtxList(testMatrix, userUIMapperTest, itemUIMapperTest, binThold); // retrieve positive user-items (rate>threshold) list from test set
    	HashMap<Integer, HashMultimap<Integer, Integer>> uciList_train=rateDao.getUserCtxList(trainMatrix, userUIMapperTrain, itemUIMapperTrain);
    	
    	int capacity = uciList.keySet().size();

    	// initialization capacity to speed up
    	List<Double> ds5 = new ArrayList<>(isDiverseUsed ? capacity : 0);
    	List<Double> ds10 = new ArrayList<>(isDiverseUsed ? capacity : 0);
    	
    	List<Double> precs1 = new ArrayList<>(capacity);
    	List<Double> precs3 = new ArrayList<>(capacity);
    	List<Double> precs5 = new ArrayList<>(capacity);
    	List<Double> precs10 = new ArrayList<>(capacity);
    	List<Double> recalls1 = new ArrayList<>(capacity);
    	List<Double> recalls3 = new ArrayList<>(capacity);
    	List<Double> recalls5 = new ArrayList<>(capacity);
    	List<Double> recalls10 = new ArrayList<>(capacity);
    	List<Double> aps = new ArrayList<>(capacity);
    	List<Double> rrs = new ArrayList<>(capacity);
    	List<Double> aucs = new ArrayList<>(capacity);
    	List<Double> ndcgs = new ArrayList<>(capacity);
    	List<Double> f1s1 = new ArrayList<>(capacity);
    	List<Double> f1s3 = new ArrayList<>(capacity);
    	List<Double> f1s5 = new ArrayList<>(capacity);
    	List<Double> f1s10 = new ArrayList<>(capacity);
    	
    	List<Double> aps1 = new ArrayList<>(capacity);
    	List<Double> aps3 = new ArrayList<>(capacity);
    	List<Double> aps5 = new ArrayList<>(capacity);
    	List<Double> aps10 = new ArrayList<>(capacity);

        // candidate items for all users: here only training items
        // use HashSet instead of ArrayList to speedup removeAll() and contains() operations: HashSet: O(1); ArrayList: O(log n).
    	Set<Integer> candItems = rateDao.getItemList(trainMatrix, itemUIMapperTrain);

    	List<String> preds = null;
    	
    	String toFile = null;
    	
    	int numTopNRanks = numRecs < 0 ? 10 : numRecs;
    	if (isResultsOut) {
    		preds = new ArrayList<String>(1500);
    		preds.add("# userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *."); // optional: file header
    		toFile = workingPath
    				+ String.format("%s-top-%d-items%s.txt", algoName, numTopNRanks, foldInfo); // the output-file name
    		FileIO.deleteFile(toFile); // delete possibly old files

    	}

    	if (verbose)
    		Logs.debug("{}{} has candidate items: {}", algoName, foldInfo, candItems.size());

    	//            // ignore items for all users: most popular items
    	//            if (numIgnore > 0) {
    	//                List<Map.Entry<Integer, Integer>> itemDegs = new ArrayList<>();
    	//                for (Integer j : candItems) {
    	//                    itemDegs.add(new SimpleImmutableEntry<Integer, Integer>(j, rateDao.getRatingCountByItem(trainMatrix,j)));
    	//                }
    	//                Lists.sortList(itemDegs, true);
    	//                int k = 0;
    	//                for (Map.Entry<Integer, Integer> deg : itemDegs) {
    	//
    	//                    // ignore these items from candidate items
    	//                    candItems.remove(deg.getKey());
    	//                    if (++k >= numIgnore)
    	//                        break;
    	//                }
    	//            }

    	// for each test user
    	
    	for (int u:uciList.keySet()) {
    		Multimap<Integer, Integer> cis = uciList.get(u);           
    		
    		int c_capacity = cis.keySet().size();

    		List<Double> c_ds5 = new ArrayList<>(isDiverseUsed ? c_capacity : 0);
    		List<Double> c_ds10 = new ArrayList<>(isDiverseUsed ? c_capacity : 0);

    		List<Double> c_precs1 = new ArrayList<>(c_capacity);
    		List<Double> c_precs3 = new ArrayList<>(c_capacity);
    		List<Double> c_precs5 = new ArrayList<>(c_capacity);
    		List<Double> c_precs10 = new ArrayList<>(c_capacity);
    		List<Double> c_recalls1 = new ArrayList<>(c_capacity);
    		List<Double> c_recalls3 = new ArrayList<>(c_capacity);
    		List<Double> c_recalls5 = new ArrayList<>(c_capacity);
    		List<Double> c_recalls10 = new ArrayList<>(c_capacity);
    		List<Double> c_aps = new ArrayList<>(c_capacity);
    		List<Double> c_rrs = new ArrayList<>(c_capacity);
    		List<Double> c_aucs = new ArrayList<>(c_capacity);
    		List<Double> c_ndcgs = new ArrayList<>(c_capacity);
    		
    		List<Double> c_aps1 = new ArrayList<>(c_capacity);
    		List<Double> c_aps3 = new ArrayList<>(c_capacity);
    		List<Double> c_aps5 = new ArrayList<>(c_capacity);
    		List<Double> c_aps10 = new ArrayList<>(c_capacity);

    		HashMultimap<Integer, Integer> cList_train = (uciList_train.containsKey(u))?uciList_train.get(u):HashMultimap.<Integer, Integer>create();

    		Multimap<Integer, Map.Entry<Integer, Double>> ciList = HashMultimap.create();
    		
    		// for each ctx
    		for (int c : cis.keySet()) {
    			if (verbose && ((u + 1) % 100 == 0))
    				Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, capacity);

    			// number of candidate items for all users
    			int numCands = candItems.size();

    			// get positive items from test matrix
    			Collection<Integer> posItems = cis.get(c);

    			List<Integer> correctItems = new ArrayList<>();

    			// intersect with the candidate items
    			for (Integer j : posItems) {
    				
    				if (isDaVIBest) {
        				if (candItems.contains(j) && !this.itemVirtualIdsTrain.containsKey(j))
        					correctItems.add(j);    					
    				}

    			}

    			if (correctItems.size() == 0)
    				continue; // no testing data for user u

    			// remove rated items from candidate items
    			Set<Integer> ratedItems = (cList_train.containsKey(c))?cList_train.get(c):new HashSet<Integer>();

    			// predict the ranking scores (unordered) of all candidate items
    			List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));
    			for (Integer j : candItems) {
    				if (!ratedItems.contains(j)) {
    					
    					if(isUserSplitting)
    						u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
    					if(isItemSplitting)
    						j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;
    					
    					double rank;

    					if (this.algoName.matches("ItemKNNUnary") || this.algoName.matches("UserKNNUnary") || this.algoName.matches("DaVIBest-ItemKNNUnary") )
    						rank = ranking(u, j, c, ratedItems);
    					else
    						rank = ranking(u, j, c);
    					
    					if(this.shouldUseItemFrequency){
    						
    						int freq = this.itemFrequency.get(j);
    						double oldRank = rank;
    						rank = rank * (1 + (freq / (this.freqNumItems - freq)));
    						
    						Logs.info("Old rank: {}, New Rank: {}", oldRank, rank);
    					}
    					

    					if (!Double.isNaN(rank)) {
    						// add rating threshold as a filter
    						if(rank>binThold) {
    							//itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
    							
    		    				if (isDaVIBest) {
    		        				if(!this.itemVirtualIdsTrain.containsKey(j))
    		        					itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
    		    				}

    						}
    					}
    				} else {
    					numCands--;
    				}
    			}

    			if (itemScores.size() == 0)
    				continue; // no recommendations available for user u

    			// order the ranking scores from highest to lowest: List to preserve orders
    			Lists.sortList(itemScores, true);
    			List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || itemScores.size() <= numRecs) ? itemScores
    					: itemScores.subList(0, numRecs);
    			
    			List<Integer> rankedItems = new ArrayList<>();
    			StringBuilder sb = new StringBuilder();
    			int count = 0;
    			for (Map.Entry<Integer, Double> kv : recomd) {
    				Integer item = kv.getKey();
    				rankedItems.add(item);

    				if (isResultsOut && count < numTopNRanks) {
    					// restore back to the original item id
    					sb.append("(").append(rateDao.getItemId(item));
    					
    					if (posItems.contains(item))
    						sb.append("*"); // indicating correct recommendation

    					sb.append(", ").append(kv.getValue().floatValue()).append(")");

    					if (++count >= numTopNRanks)
    						break;
    					if (count < numTopNRanks)
    						sb.append(", ");
    				}
    			}

    			int numDropped = numCands - rankedItems.size();
    			double AUC = Measures.AUC(rankedItems, correctItems, numDropped);
    			double AP = Measures.AP(rankedItems, correctItems);
    			double nDCG = Measures.nDCG(rankedItems, correctItems);
    			double RR = Measures.RR(rankedItems, correctItems);
    			
    			// calculate MAP@1, MAP@3, MAP@5, MAP@10
    			double AP1 = Measures.AP(rankedItems.size() <= 1 ? rankedItems: rankedItems.subList(0, 1), correctItems);
    			double AP3 = Measures.AP(rankedItems.size() <= 3 ? rankedItems: rankedItems.subList(0, 3), correctItems);
    			double AP5 = Measures.AP(rankedItems.size() <= 5 ? rankedItems: rankedItems.subList(0, 5), correctItems);
    			double AP10 = Measures.AP(rankedItems.size() <= 10 ? rankedItems: rankedItems.subList(0, 10), correctItems);


    			List<Integer> cutoffs = Arrays.asList(1, 3, 5, 10);
    			Map<Integer, Double> precs = Measures.PrecAt(rankedItems, correctItems, cutoffs);
    			Map<Integer, Double> recalls = Measures.RecallAt(rankedItems, correctItems, cutoffs);

    			c_precs1.add(precs.get(1));
    			c_precs3.add(precs.get(3));
    			c_precs5.add(precs.get(5));
    			c_precs10.add(precs.get(10));
    			c_recalls1.add(recalls.get(1));
    			c_recalls3.add(recalls.get(3));
    			c_recalls5.add(recalls.get(5));
    			c_recalls10.add(recalls.get(10));

    			c_aucs.add(AUC);
    			c_aps.add(AP);
    			c_rrs.add(RR);
    			c_ndcgs.add(nDCG);
    			
    			c_aps1.add(AP1);
    			c_aps3.add(AP3);
    			c_aps5.add(AP5);
    			c_aps10.add(AP10);

    			// diversity
    			if (isDiverseUsed) {
    				double d5 = diverseAt(rankedItems, 5);
    				double d10 = diverseAt(rankedItems, 10);

    				c_ds5.add(d5);
    				c_ds10.add(d10);
    			}

    			// output predictions
    			if (isResultsOut) {
    				// restore back to the original user id
    				preds.add(rateDao.getUserId(u) + ", " + rateDao.getContextSituationFromInnerId(c) + ": " + sb.toString());
    				
    				if (preds.size() >= 1000) {
    					FileIO.writeList(toFile, preds, true);
    					preds.clear();
    				}
    				
    			}
    		} // end a context

    		
    		// calculate metrics for a specific user averaged by contexts
    		ds5.add(isDiverseUsed ? Stats.mean(c_ds5) : 0.0);
    		ds10.add(isDiverseUsed ? Stats.mean(c_ds10) : 0.0);
    		precs1.add(Stats.mean(c_precs1));
    		precs3.add(Stats.mean(c_precs3));
    		precs5.add(Stats.mean(c_precs5));
    		precs10.add(Stats.mean(c_precs10));
    		recalls1.add(Stats.mean(c_recalls1));
    		recalls3.add(Stats.mean(c_recalls3));
    		recalls5.add(Stats.mean(c_recalls5));
    		recalls10.add(Stats.mean(c_recalls10));
    		aucs.add(Stats.mean(c_aucs));
    		ndcgs.add(Stats.mean(c_ndcgs));
    		aps.add(Stats.mean(c_aps));
    		rrs.add(Stats.mean(c_rrs));
    		
    		aps1.add(Stats.mean(c_aps1));
    		aps3.add(Stats.mean(c_aps3));
    		aps5.add(Stats.mean(c_aps5));
    		aps10.add(Stats.mean(c_aps10));
    		
    		// calculate F1 metric
    		if (Stats.mean(c_precs1)!=0 && Stats.mean(c_recalls1)!=0)
    			f1s1.add(2 * ((Stats.mean(c_precs1)*Stats.mean(c_recalls1)) / (Stats.mean(c_precs1)+Stats.mean(c_recalls1))));
    		else
    			f1s1.add(0.0);

    		if (Stats.mean(c_precs3)!=0 && Stats.mean(c_recalls3)!=0)
    			f1s3.add(2 * ((Stats.mean(c_precs3)*Stats.mean(c_recalls3)) / (Stats.mean(c_precs3)+Stats.mean(c_recalls3))));
    		else
    			f1s3.add(0.0);

    		
    		if (Stats.mean(c_precs5)!=0 && Stats.mean(c_recalls5)!=0)
    			f1s5.add(2 * ((Stats.mean(c_precs5)*Stats.mean(c_recalls5)) / (Stats.mean(c_precs5)+Stats.mean(c_recalls5))));
    		else
    			f1s5.add(0.0);
    		
    		if (Stats.mean(c_precs10)!=0 && Stats.mean(c_recalls10)!=0)
    			f1s10.add(2 * ((Stats.mean(c_precs10)*Stats.mean(c_recalls10)) / (Stats.mean(c_precs10)+Stats.mean(c_recalls10))));
    		else
    			f1s10.add(0.0);
    		
    	}

    	// write results out first
    	if (isResultsOut && preds.size() > 0) {
    		FileIO.writeList(toFile, preds, true);
    		Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
    	}

    	// measure the performance
    	Map<Measure, Double> measures = new HashMap<>();
    	measures.put(Measure.D5, isDiverseUsed ? Stats.mean(ds5) : 0.0);
    	measures.put(Measure.D10, isDiverseUsed ? Stats.mean(ds10) : 0.0);
    	measures.put(Measure.Pre1, Stats.mean(precs1));
    	measures.put(Measure.Pre3, Stats.mean(precs3));
    	measures.put(Measure.Pre5, Stats.mean(precs5));
    	measures.put(Measure.Pre10, Stats.mean(precs10));
    	measures.put(Measure.Rec1, Stats.mean(recalls1));
    	measures.put(Measure.Rec3, Stats.mean(recalls3));
    	measures.put(Measure.Rec5, Stats.mean(recalls5));
    	measures.put(Measure.Rec10, Stats.mean(recalls10));
    	measures.put(Measure.AUC, Stats.mean(aucs));
    	measures.put(Measure.NDCG, Stats.mean(ndcgs));
    	measures.put(Measure.MAP, Stats.mean(aps));
    	measures.put(Measure.MRR, Stats.mean(rrs));
    	measures.put(Measure.F11, Stats.mean(f1s1));
    	measures.put(Measure.F13, Stats.mean(f1s3));
        measures.put(Measure.F15, Stats.mean(f1s5));
        measures.put(Measure.F110, Stats.mean(f1s10));
        measures.put(Measure.MAP1, Stats.mean(aps1));
        measures.put(Measure.MAP3, Stats.mean(aps3));
        measures.put(Measure.MAP5, Stats.mean(aps5));
        measures.put(Measure.MAP10, Stats.mean(aps10));


    	return measures;

    }
    
    /** we added a verification when we use the prediction functions from itemknn and userknn unaries. 
     * @return the evaluation results of ranking predictions
     */


    protected Map<Measure, Double> evalRankings() throws Exception {
        HashMap<Integer, HashMultimap<Integer, Integer>> uciList=rateDao.getUserCtxList(testMatrix, binThold); // retrieve positive user-items (rate>threshold) list from test set
        HashMap<Integer, HashMultimap<Integer, Integer>> uciList_train=rateDao.getUserCtxList(trainMatrix);
        int capacity = uciList.keySet().size();
        
        // initialization capacity to speed up
        List<Double> ds5 = new ArrayList<>(isDiverseUsed ? capacity : 0);
        List<Double> ds10 = new ArrayList<>(isDiverseUsed ? capacity : 0);
        List<Double> precs1 = new ArrayList<>(capacity);
        List<Double> precs3 = new ArrayList<>(capacity);
        List<Double> precs5 = new ArrayList<>(capacity);
        List<Double> precs10 = new ArrayList<>(capacity);
        List<Double> recalls1 = new ArrayList<>(capacity);
        List<Double> recalls3 = new ArrayList<>(capacity);
        List<Double> recalls5 = new ArrayList<>(capacity);
        List<Double> recalls10 = new ArrayList<>(capacity);
        List<Double> aps = new ArrayList<>(capacity);
        List<Double> rrs = new ArrayList<>(capacity);
        List<Double> aucs = new ArrayList<>(capacity);
        List<Double> ndcgs = new ArrayList<>(capacity);
        List<Double> f1s1 = new ArrayList<>(capacity);
        List<Double> f1s3 = new ArrayList<>(capacity);
        List<Double> f1s5 = new ArrayList<>(capacity);
    	List<Double> f1s10 = new ArrayList<>(capacity);
    	List<Double> aps1 = new ArrayList<>(capacity);
    	List<Double> aps3 = new ArrayList<>(capacity);
    	List<Double> aps5 = new ArrayList<>(capacity);
    	List<Double> aps10 = new ArrayList<>(capacity);

        // candidate items for all users: here only training items
        // use HashSet instead of ArrayList to speedup removeAll() and contains() operations: HashSet: O(1); ArrayList: O(log n).
        Set<Integer> candItems = rateDao.getItemList(trainMatrix);

        List<String> preds = null;
        String toFile = null;
        
        int numTopNRanks = numRecs < 0 ? 10 : numRecs;
        if (isResultsOut) {
            preds = new ArrayList<String>(1500);
            preds.add("# userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *."); // optional: file header
            toFile = workingPath
                    + String.format("%s-top-%d-items%s.txt", algoName, numTopNRanks, foldInfo); // the output-file name
            FileIO.deleteFile(toFile); // delete possibly old files
            
        }

        if (verbose)
            Logs.debug("{}{} has candidate items: {}", algoName, foldInfo, candItems.size());

        // ignore items for all users: most popular items
        if (numIgnore > 0) {
            List<Map.Entry<Integer, Integer>> itemDegs = new ArrayList<>();
            for (Integer j : candItems) {
                itemDegs.add(new SimpleImmutableEntry<Integer, Integer>(j, rateDao.getRatingCountByItem(trainMatrix,j)));
            }
            Lists.sortList(itemDegs, true);
            int k = 0;
            for (Map.Entry<Integer, Integer> deg : itemDegs) {

                // ignore these items from candidate items
                candItems.remove(deg.getKey());
                if (++k >= numIgnore)
                    break;
            }
        }
        
        // for each test user
        
        for (int u:uciList.keySet()) {
        	
        	Multimap<Integer, Integer> cis = uciList.get(u);           

            int c_capacity = cis.keySet().size();

            List<Double> c_ds5 = new ArrayList<>(isDiverseUsed ? c_capacity : 0);
            List<Double> c_ds10 = new ArrayList<>(isDiverseUsed ? c_capacity : 0);
            List<Double> c_precs1 = new ArrayList<>(c_capacity);
            List<Double> c_precs3 = new ArrayList<>(c_capacity);            
            List<Double> c_precs5 = new ArrayList<>(c_capacity);
            List<Double> c_precs10 = new ArrayList<>(c_capacity);            
            List<Double> c_recalls1 = new ArrayList<>(c_capacity);
            List<Double> c_recalls3 = new ArrayList<>(c_capacity);            
            List<Double> c_recalls5 = new ArrayList<>(c_capacity);
            List<Double> c_recalls10 = new ArrayList<>(c_capacity);
            List<Double> c_aps = new ArrayList<>(c_capacity);
            List<Double> c_rrs = new ArrayList<>(c_capacity);
            List<Double> c_aucs = new ArrayList<>(c_capacity);
            List<Double> c_ndcgs = new ArrayList<>(c_capacity);
            List<Double> c_aps1 = new ArrayList<>(c_capacity);
            List<Double> c_aps3 = new ArrayList<>(c_capacity);
            List<Double> c_aps5 = new ArrayList<>(c_capacity);
            List<Double> c_aps10 = new ArrayList<>(c_capacity);

            HashMultimap<Integer, Integer> cList_train = (uciList_train.containsKey(u))?uciList_train.get(u):HashMultimap.<Integer, Integer>create();

			Multimap<Integer, List<Map.Entry<Integer, Double>>> ciList_ = HashMultimap.create();
			Multimap<Integer, Map.Entry<Integer, Double>> ciList = HashMultimap.create();
				
			
            for (int c : cis.keySet()) {
                if (verbose && ((u + 1) % 100 == 0))
                    Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, capacity);

                // number of candidate items for all users
                int numCands = candItems.size();

                // get positive items from test matrix
                Collection<Integer> posItems = cis.get(c);
                
                List<Integer> correctItems = new ArrayList<>();

                // intersect with the candidate items
                for (Integer j : posItems) {
                	if (candItems.contains(j))
                		correctItems.add(j);
                }

                if (correctItems.size() == 0)
                    continue; // no testing data for user u

                // remove rated items from candidate items
                Set<Integer> ratedItems = (cList_train.containsKey(c))?cList_train.get(c):new HashSet<Integer>();

                // predict the ranking scores (unordered) of all candidate items
                List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));
                for (Integer j : candItems) {
                    if (!ratedItems.contains(j)) {

                        if(isUserSplitting)
                            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
                        if(isItemSplitting)
                            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;
                        
                         double rank;
                        
                        if (this.algoName.matches("ItemKNNUnary") || this.algoName.matches("UserKNNUnary") || this.algoName.matches("UISplitting-ItemKNNUnary") || 
                        		this.algoName.matches("UISplitting-UserKNNUnary") || this.algoName.matches("ItemSplitting-ItemKNNUnary") || this.algoName.matches("ItemSplitting-UserKNNUnary") ||
                        		this.algoName.matches("UserSplitting-ItemKNNUnary") || this.algoName.matches("UserSplitting-UserKNNUnary") )
                        	rank = ranking(u, j, c, ratedItems);
                        else
                        	rank = ranking(u, j, c);

                        if(this.shouldUseItemFrequency){
    						int freq = this.itemFrequency.get(j);
    						rank *= (double)( 1 + ((double) freq / (double)(this.freqNumItems - freq)));						
    					}
                        
                        if (!Double.isNaN(rank)) {
                            // add rating threshold as a filter
                            if(rank>binThold){}
                                itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
                        }
                    } else {
                        numCands--;
                    }
                }

                if (itemScores.size() == 0)
                    continue; // no recommendations available for user u

                // order the ranking scores from highest to lowest: List to preserve orders
                Lists.sortList(itemScores, true);
                List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || itemScores.size() <= numRecs) ? itemScores
                        : itemScores.subList(0, numRecs);

                List<Map.Entry<Integer, Double>> recomdFinal = new ArrayList<Map.Entry<Integer,Double>>(Lists.initSize(numRecs));
                
                List<Integer> rankedItems = new ArrayList<>();
                StringBuilder sb = new StringBuilder();
                int count = 0;
                for (Map.Entry<Integer, Double> kv : recomd) {
                    Integer item = kv.getKey();
                    rankedItems.add(item);
                    ciList.put(c, new SimpleImmutableEntry<Integer, Double>(kv.getKey(), kv.getValue()));

                    if (isResultsOut && count < numTopNRanks) {
                    	// restore back to the original item id
                    	sb.append("(").append(rateDao.getItemId(item));

                    	if (posItems.contains(item))
                    		sb.append("*"); // indicating correct recommendation

                    	sb.append(", ").append(kv.getValue().floatValue()).append(")");

                    	if (++count >= numTopNRanks)
                    		break;
                    	if (count < numTopNRanks)
                    		sb.append(", ");
                    }
                }
                
                ciList_.put(c, recomdFinal);
                

                int numDropped = numCands - rankedItems.size();
                double AUC = Measures.AUC(rankedItems, correctItems, numDropped);
                double AP = Measures.AP(rankedItems, correctItems);
                double nDCG = Measures.nDCG(rankedItems, correctItems);
                double RR = Measures.RR(rankedItems, correctItems);
                
    			// calculate MAP@1, MAP@3, MAP@5, MAP@10
    			double AP1 = Measures.AP(rankedItems.size() <= 1 ? rankedItems: rankedItems.subList(0, 1), correctItems);
    			double AP3 = Measures.AP(rankedItems.size() <= 3 ? rankedItems: rankedItems.subList(0, 3), correctItems);
    			double AP5 = Measures.AP(rankedItems.size() <= 5 ? rankedItems: rankedItems.subList(0, 5), correctItems);
    			double AP10 = Measures.AP(rankedItems.size() <= 10 ? rankedItems: rankedItems.subList(0, 10), correctItems);

                //List<Integer> cutoffs = Arrays.asList(5, 10);
                List<Integer> cutoffs = Arrays.asList(1, 3, 5, 10);
                Map<Integer, Double> precs = Measures.PrecAt(rankedItems, correctItems, cutoffs);
                Map<Integer, Double> recalls = Measures.RecallAt(rankedItems, correctItems, cutoffs);
                
                c_precs1.add(precs.get(1));
                c_precs3.add(precs.get(3));
                c_precs5.add(precs.get(5));
                c_precs10.add(precs.get(10));
                c_recalls1.add(recalls.get(1));
                c_recalls3.add(recalls.get(3));
                c_recalls5.add(recalls.get(5));
                c_recalls10.add(recalls.get(10));

                c_aucs.add(AUC);
                c_aps.add(AP);
                c_rrs.add(RR);
                c_ndcgs.add(nDCG);
                
                c_aps1.add(AP1);
                c_aps3.add(AP3);
                c_aps5.add(AP5);
                c_aps10.add(AP10);

                // diversity
                if (isDiverseUsed) {
                    double d5 = diverseAt(rankedItems, 5);
                    double d10 = diverseAt(rankedItems, 10);

                    c_ds5.add(d5);
                    c_ds10.add(d10);
                }

                // output predictions
                if (isResultsOut) {
                    // restore back to the original user id
                    preds.add(rateDao.getUserId(u) + ", " + rateDao.getContextSituationFromInnerId(c) + ": " + sb.toString());

                    if (preds.size() >= 1000) {
                        FileIO.writeList(toFile, preds, true);
                        preds.clear();
                    }
                    
                }
            } // end a context


            // calculate metrics for a specific user averaged by contexts
            ds5.add(isDiverseUsed ? Stats.mean(c_ds5) : 0.0);
            ds10.add(isDiverseUsed ? Stats.mean(c_ds10) : 0.0);
            precs1.add(Stats.mean(c_precs1));
            precs3.add(Stats.mean(c_precs3));
            precs5.add(Stats.mean(c_precs5));
            precs10.add(Stats.mean(c_precs10));
            recalls1.add(Stats.mean(c_recalls1));
            recalls3.add(Stats.mean(c_recalls3));
            recalls5.add(Stats.mean(c_recalls5));
            recalls10.add(Stats.mean(c_recalls10));
            aucs.add(Stats.mean(c_aucs));
            ndcgs.add(Stats.mean(c_ndcgs));
            aps.add(Stats.mean(c_aps));
            rrs.add(Stats.mean(c_rrs));
            aps1.add(Stats.mean(c_aps1));
            aps3.add(Stats.mean(c_aps3));
            aps5.add(Stats.mean(c_aps5));
            aps10.add(Stats.mean(c_aps10));

    		if (Stats.mean(c_precs1)!=0 && Stats.mean(c_recalls1)!=0)
    			f1s1.add(2 * ((Stats.mean(c_precs1)*Stats.mean(c_recalls1)) / (Stats.mean(c_precs1)+Stats.mean(c_recalls1))));
    		else
    			f1s1.add(0.0);

    		if (Stats.mean(c_precs3)!=0 && Stats.mean(c_recalls3)!=0)
    			f1s3.add(2 * ((Stats.mean(c_precs3)*Stats.mean(c_recalls3)) / (Stats.mean(c_precs5)+Stats.mean(c_recalls3))));
    		else
    			f1s3.add(0.0);

            
    		if (Stats.mean(c_precs5)!=0 && Stats.mean(c_recalls5)!=0)
    			f1s5.add(2 * ((Stats.mean(c_precs5)*Stats.mean(c_recalls5)) / (Stats.mean(c_precs5)+Stats.mean(c_recalls5))));
    		else
    			f1s5.add(0.0);
    		
    		if (Stats.mean(c_precs10)!=0 && Stats.mean(c_recalls10)!=0)
    			f1s10.add(2 * ((Stats.mean(c_precs10)*Stats.mean(c_recalls10)) / (Stats.mean(c_precs10)+Stats.mean(c_recalls10))));
    		else
    			f1s10.add(0.0);

        }
        
        
        // write results out first
        if (isResultsOut && preds.size() > 0) {
            FileIO.writeList(toFile, preds, true);
            Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
        }

        // measure the performance
        Map<Measure, Double> measures = new HashMap<>();
        measures.put(Measure.D5, isDiverseUsed ? Stats.mean(ds5) : 0.0);
        measures.put(Measure.D10, isDiverseUsed ? Stats.mean(ds10) : 0.0);
        measures.put(Measure.Pre1, Stats.mean(precs1));
        measures.put(Measure.Pre3, Stats.mean(precs3));
        measures.put(Measure.Pre5, Stats.mean(precs5));
        measures.put(Measure.Pre10, Stats.mean(precs10));
        measures.put(Measure.Rec1, Stats.mean(recalls1));
        measures.put(Measure.Rec3, Stats.mean(recalls3));
        measures.put(Measure.Rec5, Stats.mean(recalls5));
        measures.put(Measure.Rec10, Stats.mean(recalls10));
        measures.put(Measure.AUC, Stats.mean(aucs));
        measures.put(Measure.NDCG, Stats.mean(ndcgs));
        measures.put(Measure.MAP, Stats.mean(aps));
        measures.put(Measure.MRR, Stats.mean(rrs));
        measures.put(Measure.F11, Stats.mean(f1s1));
        measures.put(Measure.F13, Stats.mean(f1s3));
        measures.put(Measure.F15, Stats.mean(f1s5));
        measures.put(Measure.F110, Stats.mean(f1s10));
        measures.put(Measure.MAP1, Stats.mean(aps1));
        measures.put(Measure.MAP3, Stats.mean(aps3));
        measures.put(Measure.MAP5, Stats.mean(aps5));
        measures.put(Measure.MAP10, Stats.mean(aps10));

        return measures;
    }

    /**
     * determine whether the rating of a user-item (u, j) is used to predicted
     *
     */
    protected boolean isTestable(int u, int j) {
        String uiid=u+","+j;
        int rowid=rateDao.getUserItemId(uiid);
        switch (view) {
            case "cold-start":
                return trainMatrix.rowSize(rowid) < 5 ? true : false;
            case "all":
            default:
                return true;
        }
    }

    /**
     *
     * @param rankedItems
     *            the list of ranked items to be recommended
     * @param cutoff
     *            cutoff in the list
     *
     * @return diversity at a specific cutoff position
     */
    protected double diverseAt(List<Integer> rankedItems, int cutoff) {

        int num = 0;
        double sum = 0.0;
        for (int id = 0; id < cutoff; id++) {
            int i = rankedItems.get(id);
            SparseVector iv = trainMatrix.column(i);

            for (int jd = id + 1; jd < cutoff; jd++) {
                int j = rankedItems.get(jd);

                double corr = corrs.get(i, j);
                if (corr == 0) {
                    // if not found
                    corr = correlation(iv, trainMatrix.column(j));
                    if (!Double.isNaN(corr))
                        corrs.set(i, j, corr);
                }

                if (!Double.isNaN(corr)) {
                    sum += (1 - corr);
                    num++;
                }
            }
        }

        return 0.5 * (sum / num);
    }

    protected double ranking(int u, int j, int c) throws Exception {
        return predict(u, j, c, false);
    }
    
    protected double ranking(int u, int j, int c, Set<Integer> obs) throws Exception {
    	return predict(u, j, c, obs, false);
    }

    protected SymmMatrix buildCorrs(boolean isUser) {
        Logs.debug("Build {} similarity matrix ...", isUser ? "user" : "item");

        int count = isUser ? numUsers : numItems;
        SymmMatrix corrs = new SymmMatrix(count);

        for (int i = 0; i < count; i++) {
            SparseVector iv = isUser ? train.row(i) : train.column(i);
            if (iv.getCount() == 0)
                continue;
            // user/item itself exclusive
            for (int j = i + 1; j < count; j++) {
                SparseVector jv = isUser ? train.row(j) : train.column(j);
                if(jv.getCount() == 0 )
                    continue;
                double sim = correlation(iv, jv);

                if (!Double.isNaN(sim))
                    corrs.set(i, j, sim);
            }
        }

        return corrs;
    }

    protected SymmMatrix buildCorrs(boolean isUser, SparseMatrix train) {
        Logs.debug("Build {} similarity matrix ...", isUser ? "user" : "item");

        int count = isUser ? numUsers : numItems;
        SymmMatrix corrs = new SymmMatrix(count);

        for (int i = 0; i < count; i++) {
            SparseVector iv = isUser ? train.row(i) : train.column(i);
            if (iv.getCount() == 0)
                continue;
            // user/item itself exclusive
            for (int j = i + 1; j < count; j++) {
                SparseVector jv = isUser ? train.row(j) : train.column(j);
                if(jv.getCount() == 0 )
                    continue;
                double sim = correlation(iv, jv);

                if (!Double.isNaN(sim))
                    corrs.set(i, j, sim);
            }
        }

        return corrs;
    }


    /**
     * initilize recommender model
     */
    protected void initModel() throws Exception {
        if(isItemSplitting==false && isUserSplitting==false && isCARSRecommender==false && isDaVIBest==false && isDaVIAll==false) // return a 2D rating matrix based on original CARS data set
            train=rateDao.toTraditionalSparseMatrix(trainMatrix);

        else if(isCARSRecommender==false && isDaVIBest==false && isDaVIAll==false) // return a 2D rating matrix based on transformation from splitting mappers
            train=this.createTraditionalSparseMatrixBySplitting();
        else if (isCARSRecommender==false) // return a 2D rating matrix based on transformation from DaVI operation
        	train=this.createTraditionalSparseMatrixFromDaVI(trainMatrix, userUIMapperTrain, itemUIMapperTrain);

    }

    /**
     * Learning method: override this method to build a model, for a model-based method. Default implementation is
     * useful for memory-based methods.
     *
     */
    protected void buildModel() throws Exception {
    }

    /**
     * After learning model: release some intermediate data to avoid memory leak
     */
    protected void postModel() throws Exception {
    }

    /**
     * Serializing a learned model (i.e., variable data) to files.
     */
    protected void saveModel() throws Exception {
    }

    /**
     * Deserializing a learned model (i.e., variable data) from files.
     */
    protected void loadModel() throws Exception {
    }

    private void printAlgoConfig() {
        String algoInfo = toString();

        Class<? extends Recommender> cl = this.getClass();
        // basic annotation
        String algoConfig = cl.getAnnotation(Configuration.class).value();

        // additional algorithm-specific configuration
        if (cl.isAnnotationPresent(AddConfiguration.class)) {
            AddConfiguration add = cl.getAnnotation(AddConfiguration.class);

            String before = add.before();
            if (!Strings.isNullOrEmpty(before))
                algoConfig = before + ", " + algoConfig;

            String after = add.after();
            if (!Strings.isNullOrEmpty(after))
                algoConfig += ", " + after;
        }

        if (!algoInfo.isEmpty()) {
            if (!algoConfig.isEmpty())
                Logs.debug("{}: [{}] = [{}]", algoName, algoConfig, algoInfo);
            else
                Logs.debug("{}: {}", algoName, algoInfo);
        }
    }

    /**
     * logistic function g(x)
     */
    protected double g(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    /**
     * gradient value of logistic function g(x)
     */
    protected double gd(double x) {
        return g(x) * g(-x);
    }

    /**
     * Check if ratings have been binarized; useful for methods that require binarized ratings;
     */
    protected void checkBinary() {
        if (binThold < 0) {
            Logs.error("val.binary.threshold={}, ratings must be binarized first! Try set a non-negative value.",
                    binThold);
            System.exit(-1);
        }
    }

    public void run() {
        try {
            execute();
        } catch (Exception e) {
            // capture error message
            Logs.error(e.getMessage());

            e.printStackTrace();
        }
    }
}
