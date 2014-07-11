import java.util.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;

public class CollectionPreprocessor {

  //private final static String unkToken = "=unk=";

  private static boolean DEBUG=false; // Thang Apr14

  static class LabeledDocumentRepr {
    List<Double> repr;
    int label;

    LabeledDocumentRepr(List<Double> repr, int label) {
      this.repr = repr;
      this.label = label;
    }
    void saveDocument(BufferedWriter out) throws IOException{
      out.write(label + " ");
      int idx = 1;
      for (Double feat : repr) {
        out.write(idx + ":" + feat + " ");
        idx++;
      }
      out.newLine();
    }

  }


  private static Map<String,String> getParamMap(String args[]) {
    HashMap<String,String> name2Value = new HashMap<String, String>();

    int idx = 0;
    while (idx < args.length) {
      String key = args[idx];
      if (!key.startsWith("--")) {
        return null;
      }
      idx++;
      
      // Thang May14: allow for --flag without argument
      if (idx == args.length) {
        name2Value.put(key, "true");
        return null;
      }
      
      if(!args[idx].startsWith("--")){ // not another flag
        name2Value.put(key, args[idx]);
        idx++;
      } else { // another flag, so this one doesn't have an argument
        name2Value.put(key, "true");
      }
    }
    return name2Value;

  }

  private static void checkParams(Map<String,String> params) {

    if (!params.containsKey("--text-dir") ||
        !params.containsKey("--idf") ||
        !params.containsKey("--word-embeddings") ||
        !params.containsKey("--vector-file")) {

      System.out.println("Wrong parameters");
      System.out.println("Usage: java CollectionPreprocess --text-dir directory --idf idf-file --word-embeddings embedding-file --vector-file output [--rnnlm]");
      System.exit(-1);
    }


  }

  private static Map<String,Double> loadIdf(String idfFile) throws IOException{
    BufferedReader in = new BufferedReader(new FileReader(idfFile));
    Map<String,Double> idf = new HashMap<String, Double>(); 

    String line;
    while ((line = in.readLine()) != null) {
      line = line.trim();
      if (line.equals("")) {
        continue;
      }

      String fields[] = line.split("\\s+");
      assert fields.length == 3;
      idf.put(fields[0], Double.parseDouble(fields[2]));

    }


    in.close();
    return idf;
  }

  // add rnnmFlag to read in rnnlm word vectors
  // find out embedding dimension first
  public static Map<String,List<Double>> loadEmbeddings(String embedFile, boolean rnnlmFlag) throws IOException {
    BufferedReader in = new BufferedReader(new FileReader(embedFile));
    Map<String,List<Double>> wordEmbeddings = new HashMap<String, List<Double>>();

    String line = null;
    
    // Thang May14
    int embedDim=-1;
    if (rnnlmFlag) { // read header line 
      String[] tokens = in.readLine().trim().split("\\s+"); 
      embedDim = Integer.parseInt(tokens[1]);
      
      line = in.readLine();
      if(line==null) {
        in.close();
        throw new RuntimeException("! Empty embedding file: " + embedFile); 
      }
    } else { 
      line = in.readLine(); // read ahead a line
      if(line==null) {
        in.close();
        throw new RuntimeException("! Empty embedding file: " + embedFile); 
      }
      embedDim = line.trim().split("\\s+").length-1; 
    }
    
    // read file content
    while(line!=null) {
      line = line.trim();
      if (line.equals("")) { 
        line = in.readLine();
        continue; 
      }
      String tokens[] = line.split("\\s+");

      String word = tokens[0];
      ArrayList<Double> features;
      features = new ArrayList<Double>(embedDim);
      
      if(embedDim != (tokens.length-1)){
        in.close();
        throw new RuntimeException("Word " + word + " has different dim: " + embedDim + " != " +  (tokens.length-1));
      }
      for (int idx = 1; idx < tokens.length; idx++) {
        features.add(Double.parseDouble(tokens[idx]));
      }
      wordEmbeddings.put(word, features);

      line = in.readLine();
    }

    in.close();

    return wordEmbeddings;
  }


  static LabeledDocumentRepr computeDocumentRepr(int label, String file, Map<String,Double> idf, Map<String,List<Double>> wordEmbeds) 
      throws IOException {

    Map<String,Integer> wordCounts = new HashMap<String, Integer>();

    BufferedReader in = new BufferedReader(new FileReader(file));

    String line;
    while ((line = in.readLine()) != null) {
      line = line.trim();
      if (line.equals("")) {
        continue;
      }
      String tokens[] = line.split("\\s+");
      for (String tok : tokens) {
        tok = tok.toLowerCase();
        if (!wordCounts.containsKey(tok)) {
          wordCounts.put(tok, 0);
        }
        wordCounts.put(tok, wordCounts.get(tok) + 1);
      }
    }
    in.close();


    int embedSize = wordEmbeds.get(wordEmbeds.keySet().iterator().next()).size();
    ArrayList<Double> repr = new ArrayList<Double>(embedSize);

    for (int idx = 0; idx < embedSize; idx++) {
      repr.add(idx, 0.);
    }

    double norm = 0;
    for (String word : wordCounts.keySet()) {
      if (!idf.containsKey(word) || !wordEmbeds.containsKey(word)) {
        continue;
      }
      norm += idf.get(word);
    }


    if (norm != 0) {
      for (String word : wordCounts.keySet()) {
        if (!wordEmbeds.containsKey(word) || ! idf.containsKey(word)) {
          continue;
        }
        List<Double> vector = wordEmbeds.get(word);  // arraylist
        Double w = idf.get(word);
        for (int idx = 0; idx < embedSize; idx++) {
          double inc = w * vector.get(idx) * wordCounts.get(word) / norm; // TODO: check if 0/1 is better
          repr.set(idx, repr.get(idx) + inc);
        }
      }
    }

    return new LabeledDocumentRepr(repr, label);

  }


  public static void main(String[] args)  throws Exception {
    Random rand = new Random(0);

    if (DEBUG) { System.err.println("Command line parameters:" + Arrays.toString(args)); }

    Map<String,String> key2Value = getParamMap(args);

    if (DEBUG) { System.err.println("Command line parameters:" + key2Value); }

    checkParams(key2Value);

    String idfFile = key2Value.get("--idf");
    if (DEBUG) { System.err.println("Reading the IDF file..."); }
    Map<String,Double> idf = loadIdf(idfFile);

    String wordEmbedFile = key2Value.get("--word-embeddings");
    if (DEBUG) { System.err.println("Reading the word embedding file..."); }
    
    // Thang May14: add rnnlm flag
    boolean rnnlmFlag = false;
    if(key2Value.containsKey("--rnnlm")) { rnnlmFlag = true; }
    Map<String,List<Double>> wordEmbeds = loadEmbeddings(wordEmbedFile, rnnlmFlag);


    String dataDir =  key2Value.get("--text-dir");
    File labelDir = new File(dataDir);

    String labels[] = labelDir.list();
    Arrays.sort(labels);
    Map<String,Integer> label2Id = new HashMap<String, Integer>();
    for (int idx = 0; idx < labels.length; idx++) {
      label2Id.put(labels[idx], idx + 1);
    }

    List<LabeledDocumentRepr> reprs = new LinkedList<CollectionPreprocessor.LabeledDocumentRepr>(); 

    if (DEBUG) { System.out.println("Reading and processing the text files..."); }
    for (String label : labels) {
      File d = new File(labelDir.getPath() + "/" +  label);

      if (!d.isDirectory())
        continue;

      String[] files = d.list();
      Arrays.sort(files);

      // file.indexOf is replaced with file.lastIndexOf in order to deal
      // with dot char in the filename -- for paired data
      for (String file : files) {
        String fileName = d.getPath() + "/" + file;
        LabeledDocumentRepr docRepr = computeDocumentRepr(label2Id.get(label), fileName, idf, wordEmbeds);
        reprs.add(docRepr);
      }
    }



    Collections.shuffle(reprs, rand);
    if (DEBUG) { System.err.println("Shuffled the documents"); }

    BufferedWriter out = new BufferedWriter(new FileWriter(key2Value.get("--vector-file")));

    if (DEBUG) { System.err.print("Saving the produced document representations..."); }

    for (LabeledDocumentRepr doc : reprs) {
      doc.saveDocument(out);
    }
    out.close();

    if (DEBUG) { System.err.println("done."); }
  }
}
