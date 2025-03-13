import static org.junit.Assert.*;
import org.junit.*;

public class ConverterTest {
    
    @Test
    public void invalidChar(){
        try{
            Converter.strToInt("23f");
            fail();
        }
        catch(NumberFormatException e){
            //ok
        }
    }
}
