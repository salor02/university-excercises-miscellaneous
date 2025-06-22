CREATE FUNCTION prelievo (prod INTEGER, quant INTEGER) RETURNS INTEGER AS $$
DECLARE
	new_qty INTEGER;
BEGIN
	UPDATE magazzino
	SET qta_disp = qta_disp - quant
	WHERE cod_prod = prod;
	
	SELECT qta_disp
	INTO new_qty
	FROM magazzino
	WHERE cod_prod = prod;
	
	RETURN new_qty;
END;

$$ LANGUAGE 'plpgsql';