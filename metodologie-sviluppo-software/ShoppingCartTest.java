import static org.junit.Assert.*;
import org.junit.*;

public class ShoppingCartTest{
    private ShoppingCart cart;

    @Before
    public void setUp(){
        cart = new ShoppingCart();
    }

    @Test
    public void testCreated(){
        assertEquals(cart.getItemCount(), 0);
    }

    @Test
    public void testEmpty(){
        cart.empty();
        assertEquals(cart.getItemCount(), 0);
    }

    @Test
    public void testAdd(){
        int prev = cart.getItemCount();
        cart.addItem(new Product("prova", 12.5));
        assertEquals(cart.getItemCount(), prev + 1);
    }

    @Test
    public void testAddBalance(){
        cart.addItem(new Product("prova1", 35.5));
        double prevBalance = cart.getBalance();
        cart.addItem(new Product("prova2", 12.5));
        assertEquals(cart.getBalance(), prevBalance + 12.5, 0.00001);
    }

    @Test
    public void testRemove(){
        //dummy items
        cart.addItem(new Product("prova1", 126.5));
        cart.addItem(new Product("prova2", 124.5));

        //item to remove
        Product item = new Product("prova3", 122.5);
        cart.addItem(item);

        int prev = cart.getItemCount();

        try{
            cart.removeItem(item);
            assertEquals(cart.getItemCount(), prev - 1);
        }
        catch(ProductNotFoundException e){
            //it never happens here
        }
    }

    @Test
    public void testExceptionOnRemove(){
        //dummy items
        cart.addItem(new Product("prova1", 126.5));
        cart.addItem(new Product("prova2", 124.5));

        //this item is not added
        Product fakeItem = new Product("fake", 12);

        try{
            cart.removeItem(fakeItem);
            fail();
        }
        catch(ProductNotFoundException e){
            //exception successfull raised
        }
    }

    
}
