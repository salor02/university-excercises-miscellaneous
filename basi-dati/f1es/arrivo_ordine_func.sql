CREATE FUNCTION arrivo_ordine(prod INTEGER) RETURNS INTEGER AS $$
DECLARE
	q_riord INTEGER;
	new_qta INTEGER;
BEGIN
	SELECT qta_ord INTO q_riord
	FROM riordino
	WHERE cod_prod = prod;
	
	IF(q_riord > 0) THEN
		UPDATE magazzino
		SET qta_disp = qta_disp + q_riord
		WHERE cod_prod = prod;
		
		SELECT qta_disp INTO new_qta
		FROM magazzino
		WHERE cod_prod = prod;
		
		DELETE FROM riordino WHERE cod_prod = prod;
	ELSE
		RAISE EXCEPTION 'Prodotto non in riordine';
	END IF;
	
	RETURN new_qta;
END;
$$ LANGUAGE 'plpgsql';