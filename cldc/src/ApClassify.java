import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class ApClassify {
  private static boolean DEBUG=false; // Thang Apr14
  
  private static Map<String,String> getParamMap(String args[]) {
    HashMap<String,String> name2Value = new HashMap<String, String>();

    int idx = 0;
    while (idx < args.length) {
      String key = args[idx];
      if (!key.startsWith("--")) {
        return null;
      }
      idx++;
      if (idx == args.length) {
        return null;
      }
      name2Value.put(key, args[idx]);
      idx++;
    }
    return name2Value;

  }


  private static void printUsageExit() {
    System.out.println("Usage: java ApClassify --test-set [file name] --model-name [file name]");
  }
  /**
   * @param args
   */
  public static void main(String[] args) throws  IOException {
    Map<String,String> paramMap = getParamMap(args);
    if (paramMap.size() == 0) {
      printUsageExit();
    }

    if(DEBUG) { System.err.print("Params: " + paramMap + "  "); }


    String fileName = paramMap.get("--test-set");
    String modelName = paramMap.get("--model-name");

    InputReader reader = new InputReader();
    List<Example> data = reader.readData(fileName);

    Model mod = Model.load(modelName);

    int correct = 0;
    for (Example ex : data) {
      int pred = mod.predictLabel(ex.sv);
      if (pred == ex.label) {
        correct++;
      }
    }
    //System.err.println("  accuracy: " + (((double) correct) / data.size()));
    System.err.format(" %2.2f\n", (((double) correct*100) / data.size()));

  }

}
