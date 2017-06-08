package carskit.data.structure;

import java.util.ArrayList;
import java.util.List;

import carskit.generic.Recommender;
import carskit.generic.Recommender.Measure;

public class Segment {

	private SparseMatrix 		data;
	private String 		 		name;
	private Recommender			recommender;
	private final List<Double> 	metrics = new ArrayList<>();
	private final Integer 		condition;
		
	public Segment(SparseMatrix data, String name, Integer conditionKey) {
		this.data = data;
		this.name = name;
		this.condition = conditionKey;
	}
	
	public SparseMatrix getData() {
		return data;
	}

	public String getName() {
		return name;
	}

	public void addMetric(Double v){
		metrics.add(v);
	}
	
	public Double getMetric(){
		return (metrics.stream().reduce(0.0, Double::sum)) / metrics.size();
	}

	public Recommender getRecommender() {
		return recommender;
	}
	
	public void setRecommender(Recommender rec) {
		this.recommender = rec;
	}

	public Integer getCondition() {
		return condition;
	}
	
	public void updateMetrics (double metric){
		this.metrics.removeAll(this.metrics);
		this.metrics.add(metric);
	}
	
	

}
