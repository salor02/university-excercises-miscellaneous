public class Converter {
    public static int strToInt(String stringNumber){
        try{
            int intNumber = Integer.parseInt(stringNumber);
            return intNumber;
        }
        catch(NumberFormatException e){
            throw e;
        }
    }
}
