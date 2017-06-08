package carskit.alg.cars.transformation.virtualitems;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Set;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;

import carskit.data.processor.DataTransformer;
import happy.coding.io.FileIO;

public class ToTraditionalData {
	
	/* a pre-process step
	 * 1) make an early DaVI-transformation from multidimensional compact format to two-dimensional compact format
	 *    note: in the resulting data, each item marked with tag 'd' means virtual item
	 * 
	 * 2) make a dataset transformation from compact format to binary format (CARSKit default format)
	 * 
	 * General note:
	 * Each file data resulting will be used by DaVI implementation in CARSKit
	 */

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String WorkingPath = "/home/raoni/workspace/CARSKit/context-aware_data_sets/toy/CARSKit.Workspace/";
		String OriginalRatingDataPath = "/home/raoni/workspace/CARSKit/context-aware_data_sets/toy/toy_movie2.txt";
		String OutputFolder = "/home/raoni/workspace/CARSKit/context-aware_data_sets/toy/";
		BufferedReader br = FileIO.getReader(OriginalRatingDataPath);
		String line = br.readLine(); // 1st line is header
		
		String[] header = line.split(",",-1);
			
        int dimscount = header.length - 3;
        String[] dims = new String[dimscount];
        for (int i = 3; i < header.length; ++i)
            dims[i - 3] = header[i].trim().toLowerCase();
		
		String[] data = line.trim().split(",", -1);
		
		HashMap<String, HashMap<String, String>> newlines = new LinkedHashMap();
		Multimap<String, String> conditions = TreeMultimap.create(); // key=dim, value=cond, keep the order when we adding to it
		
		while ((line = br.readLine()) != null) {
			data = line.trim().split(",",-1);
			String user = data[0];
			String item = data[1];
			Double rate = Double.valueOf(data[2]);
			
            String[] strs = line.split(",", -1);
            HashMap<String, String> ratingcontext = new HashMap<>();
            for (int i=3 ; i<3+dimscount ; ++i) {
            	String cond = strs[i];
            	ratingcontext.put(dims[i -3], cond);
            	conditions.put(dims[i - 3], cond);
            }
			
            newlines.put(user + "," + item + "," + rate, ratingcontext);			
		}
		br.close();
		
        // create new header
		
        StringBuilder headerBuilder = new StringBuilder();
        headerBuilder.append("userid,itemid,rating");
        
        for (String dim : conditions.keySet()) {
        	if (headerBuilder.length() > 0) headerBuilder.append(",");
        	headerBuilder.append(dim);
        }
		
        BufferedWriter bw = FileIO.getWriter(OutputFolder + "toy_movie2_extended.txt");
        bw.write(headerBuilder.toString() + "\n");
        bw.flush();       
        
        StringBuilder recordBuilder = new StringBuilder();
        
		for (String uir : newlines.keySet()) {
			String[] uir_split = uir.split(",",-1);
			recordBuilder.append(uir_split[0] + "," + uir_split[1] + "," + uir_split[2]);
			
			HashMap<String,String> uiConds = new HashMap<String,String>();
			Set <String> ds = newlines.get(uir).keySet();
			
			for (String d : ds) {
				recordBuilder.append("," + newlines.get(uir).get(d));
				uiConds.put(d, newlines.get(uir).get(d));
			}
			recordBuilder.append("\n");
			
			for (String d1 : ds) { // create virtual items for each conditional context
				recordBuilder.append(uir_split[0] + "," + d1 + ":" + newlines.get(uir).get(d1) + "," + uir_split[2]);
				for (String d2 : ds)
					recordBuilder.append("," + uiConds.get(d2));
				recordBuilder.append("\n");
			}
		}
		
		bw.write(recordBuilder.toString());
		bw.flush();
		
		bw.close();
		
		// create a file with conditions values in each dimension 
		       
        // we add missing values to the condition sets in the same way as used in CARSKit
        for (String dim : conditions.keySet()) {
            conditions.put(dim, "na");
        }
        
        StringBuilder strConditionsBuilder = new StringBuilder();

        
        for (String dim : conditions.keySet()) {
            Collection<String> conds = conditions.get(dim);
            for (String cond : conds) {
                if (strConditionsBuilder.length() > 0) strConditionsBuilder.append(",");
                strConditionsBuilder.append(dim + ":" + cond);
            }
        }
        
        BufferedWriter bw2 = FileIO.getWriter(OutputFolder + "toy_movie2_conditions.txt");
        bw2.write(strConditionsBuilder.toString());
        bw2.flush();
        bw2.close();

        //bw2.write(headerBuilder.toString() + "\n");
        //bw2.flush();
		
		
        //DataTransformer transformer = new DataTransformer();
        
        //int flag = validateDataFormat(OriginalRatingDataPath);
        //transformer.setParameters(flag, OriginalRatingDataPath, WorkingPath);
        //Thread t = new Thread(transformer);
        //t.start();
        //t.join(); // for large data, the transformation and output to external file may take time!


	}
	
	protected static boolean isBinaryNumber(int number) { int copyOfInput = number; while (copyOfInput != 0) { if (copyOfInput % 10 > 1) { return false; } copyOfInput = copyOfInput / 10; } return true; }
	
    protected static int validateDataFormat(String dataPath) throws Exception {

        // flag=0, did not check; flag=1, binary format; flag=2, loose format; flag=3, compact format
        int flag=0;
        BufferedReader br = FileIO.getReader(dataPath);
        String header = br.readLine(); // 1st line;
        String dataline = br.readLine();
        br.close();

        String[] sheader=header.split(",",-1);
        String[] sdata=dataline.split(",",-1);
        String lastColumn=sheader[sheader.length-1].trim().toLowerCase();
        if(sheader[sheader.length-2].trim().toLowerCase().equals("dimension") &&
                lastColumn.equals("condition"))
        {
            flag=2; // it is loose format
        }else
        {
            boolean isBinary=true; // header is as "dimension:condition"
            for(int i=3;i<sheader.length;++i)
            {
                if(sheader[i].indexOf(":")==-1 || isBinaryNumber(Integer.valueOf(sdata[i]))==false)
                {
                    isBinary=false;
                    break;
                }
            }
            if(isBinary) // basically, it is probably binary format
            {
                flag=1; // it is binary format
            }else
            {
                flag=3; // it is compact format
            }
        }
        return flag; // those are very basic validations; if there are data errors, the readData() will throw errors
    }


}
